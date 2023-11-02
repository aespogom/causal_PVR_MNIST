import argparse
import json
import random
import torch
from torch import nn
from torch.optim import AdamW

import math
import os
import time
import shutil

from tqdm import tqdm
from counterfactual_utils import deserialize_variable_name, parse_variable_name, get_activation_at
from dataset.BlockStylePVR import BlockStylePVR, setup_loaders
from models.student import resnet18 as student
from models.teacher import oracle as teacher
from utils import logger, set_seed
from transformers import get_linear_schedule_with_warmup


## TRAINER
class Trainer:
    def __init__(
        self,
        params: dict,
        dataset: BlockStylePVR, 
        neuro_mapping: str,
        student: nn.Module,
        teacher: nn.Module
    ):
        logger.info("Initializing Trainer")
        self.params = params
        self.dump_path = params.dump_path
        self.neuro_mapping = neuro_mapping
        self.student = student
        self.teacher = teacher

        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        
        # causal neuron mappings.
        self.deserialized_interchange_variable_mappings = []
        with open(self.neuro_mapping) as json_file:
            neuron_mapping = json.load(json_file)
            logger.info(f"Neuron Mapping: {neuron_mapping}")
            interchange_variable_mappings = neuron_mapping["interchange_variable_mappings"]
            for m in interchange_variable_mappings:
                teacher_deserialized_variables = []
                for variable in m["teacher_variable_names"]:
                    teacher_deserialized_variables.append(deserialize_variable_name(variable))
                student_deserialized_variables = []
                for variable in m["student_variable_names"]:
                    student_deserialized_variables.append(deserialize_variable_name(variable))
                self.deserialized_interchange_variable_mappings += [
                    [teacher_deserialized_variables, student_deserialized_variables]
                ]
        logger.info(f"Deserialized interchange variable mappings {str(self.deserialized_interchange_variable_mappings)}.")

        self.dataloader = dataset

        self.loss = nn.CrossEntropyLoss()

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 1.0e-5,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=0.05, eps=1e-06, betas=(0.9, 0.98)
        )

        warmup_steps = math.ceil(num_train_optimization_steps * 0.05)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )

    def train(self):
        """
        The real training loop.
        """
        logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
        self.teacher.eval()

        for _ in range(self.params.n_epoch):
            logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")

            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                x, value = batch

                if self.params.n_gpu > 0:
                    # x = x.to(f"cuda:0", non_blocking=True)
                    # value = value.to(f"cuda:0", non_blocking=True)
                    pass

                self.step(
                    input_ids=x,
                    labels=value
                )
                iter_bar.update()
                iter_bar.set_postfix(
                    {
                        "Last_loss": f"{self.last_loss:.2f}", 
                        "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}"
                    }
                )
            iter_bar.close()

            logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        logger.info("Save very last checkpoint as `pytorch_model.pth`.")
        self.save_checkpoint(checkpoint_name="pytorch_model.pth")
        logger.info("Training is finished")

    def step(
        self,
        input_ids: torch.tensor, 
        labels: torch.tensor
    ):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).
        Input:
        ------
       """
        # we randomly select the pool of neurons to interchange.
        selector = random.randint(0, len(self.deserialized_interchange_variable_mappings)-1)
        interchange_variable_mapping = self.deserialized_interchange_variable_mappings[selector]
        teacher_variable_names = random.choice(interchange_variable_mapping[0])
        logger.info(f"teacher_variable_names {str(teacher_variable_names)}.")
        student_variable_names = random.choice(interchange_variable_mapping[1])
        logger.info(f"student_variable_names {str(student_variable_names)}.")
        teacher_interchanged_variables_mapping = {}
        student_interchanged_variables_mapping = {}
        # we need to do the interchange here.
        for i, variable in enumerate(teacher_variable_names):
            layer_index, head_index, LOC = parse_variable_name(variable)
            if layer_index in teacher_interchanged_variables_mapping:
                teacher_interchanged_variables_mapping[layer_index] += [(i, head_index, LOC)]
            else:
                teacher_interchanged_variables_mapping[layer_index] = [(i, head_index, LOC)]
        for i, variable in enumerate(student_variable_names):
            layer_index, head_index, LOC = parse_variable_name(variable)
            if layer_index in student_interchanged_variables_mapping:
                student_interchanged_variables_mapping[layer_index] += [(i, head_index, LOC)]
            else:
                student_interchanged_variables_mapping[layer_index] = [(i, head_index, LOC)]
        logger.info(f"student_interchanged_variables_mapping {str(student_interchanged_variables_mapping)}.")
        logger.info(f"teacher_interchanged_variables_mapping {str(teacher_interchanged_variables_mapping)}.")
        
        counterfactual_input_ids = input_ids
        
        with torch.no_grad():
            # teacher forward pass normal.
            teacher_outputs = self.teacher(
                input_ids=input_ids, labels=labels
            )
            # teacher forward pass for interchange variables.
            
            dual_counterfactual_activations_teacher = get_activation_at(
                self.teacher,
                input_ids=input_ids, # this is different!
                variable_names=teacher_variable_names
            )
            # logger.info(f"dual_counterfactual_activations_teacher {str(dual_counterfactual_activations_teacher)}.")
            # teacher forward pass for interchanged outputs.
            counterfactual_outputs_teacher = self.teacher(
                input_ids=counterfactual_input_ids, # this is different!
                labels=labels,
                interchanged_variables=dual_counterfactual_activations_teacher,
                variable_names=teacher_interchanged_variables_mapping,
            )
            # logger.info(f"counterfactual_outputs_teacher {str(counterfactual_outputs_teacher)}.")

        t_logits, t_hidden_states = \
            teacher_outputs["logits"], teacher_outputs["hidden_states"]
        student_outputs = self.student(
            input_ids=input_ids,
            labels=labels,
            t_logits=t_logits,
            t_hidden_states=t_hidden_states,
        )  # (bs, seq_length, voc_size)
        s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
        causal_t_logits, causal_t_hidden_states = \
            counterfactual_outputs_teacher["logits"], counterfactual_outputs_teacher["hidden_states"]
        #falta sumar losss de aqui
        # we need to get causal distillation loss!
        dual_counterfactual_activations_student = get_activation_at(
            self.student,
            input_ids, # this is different!
            variable_names=student_variable_names
        )
        # dual on main.
        counterfactual_outputs_student = self.student(
            input_ids=counterfactual_input_ids, # this is different!
            labels=labels,
            # intervention
            interchanged_variables=dual_counterfactual_activations_student,
            variable_names=teacher_interchanged_variables_mapping,
            # loss.
            t_logits=t_logits,
            t_hidden_states=t_hidden_states,
            causal_t_logits=causal_t_logits,
            causal_t_hidden_states=causal_t_hidden_states,
            s_logits=s_logits,
            s_hidden_states=s_hidden_states,
        )
                
        self.total_loss_epoch += counterfactual_outputs_student["loss"].item()
        self.last_loss = counterfactual_outputs_student["loss"].item()
            
        self.optimize(counterfactual_outputs_student["loss"])

        self.n_sequences_epoch += input_ids.size(0)

    def optimize(self, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        loss.backward()

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))

def prepare_trainer(args):

    # ARGS #
    set_seed(args)
    # More validations #
    
    # if not args.force:
    #     raise ValueError(
    #         f"Serialization dir {args.dump_path} already exists, but you have not precised wheter to overwrite it"
    #         "Use `--force` if you want to overwrite it"
    #     )
    # else:
    shutil.rmtree(args.dump_path)

    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)
    logger.info(f"Experiment will be dumped and logged in {args.dump_path}")

    # SAVE PARAMS #
    logger.info(f"Param: {args}")
    with open(os.path.join(args.dump_path, "parameters.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    student_model = student.ResNet18()
    # student = student_model.to(f"cuda:0", non_blocking=True)
    logger.info("Student loaded.")
    teacher_model = teacher.Oracle()
    # teacher = teacher_model.to(f"cuda:0", non_blocking=True)
    logger.info("Teacher loaded.")

    # DATA LOADER
    train_dataset, _ = setup_loaders()
    logger.info("Data loader created.")

    # TRAINER #
    torch.cuda.empty_cache()
    trainer = Trainer(
        params=args,
        dataset=train_dataset,
        neuro_mapping=args.neuro_mapping,
        student=student_model,
        teacher=teacher_model
    )
    logger.info("trainer initialization done.")
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite dump_path if it already exists."
    )

    parser.add_argument(
        "--dump_path",
        type=str,
        default="results",
        help="The output directory (log, checkpoints, parameters, etc.)"
    )
    parser.add_argument(
        "--neuro_mapping",
        type=str,
        default="training_configs/MNIST.nm",
        help="Predefined neuron mapping for the interchange experiment.",
    )
    parser.add_argument("--n_epoch", type=int, default=3, help="Number of pass on the whole dataset.")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size (for each process).")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=50,
        help="Gradient accumulation for larger training batches.",
    )
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs in the node.")
    parser.add_argument("--local_rank", type=int, default=0, help="Distributed training - Local rank")
    parser.add_argument("--seed", type=int, default=56, help="Random seed")

    parser.add_argument("--log_interval", type=int, default=500, help="Tensorboard logging interval.")
    parser.add_argument("--checkpoint_interval", type=int, default=4000, help="Checkpoint interval.")
    
    args = parser.parse_args()
    
    # config the runname here and overwrite.
    run_name = f"s_resnet18_t_oracle_data_MNIST_seed_{args.seed}"
    args.run_name = run_name
    args.dump_path = os.path.join(args.dump_path, args.run_name)
    trainer = prepare_trainer(args)
    logger.info("Start training.")
    trainer.train()


