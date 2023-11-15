import argparse
import itertools
import json
import random
import torch
from torch import nn
from torch.optim import AdamW

import math
import os
import time
import shutil
import numpy as np

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
        val_dataset: BlockStylePVR,
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
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.total_ii_acc_epoch = 0
        self.last_ii_acc = 0
        self.total_beh_acc_epoch = 0
        self.last_beh_acc = 0
        
        # Deserialize causal neuron mappings
        # $L:X$H:Y$[Z:Z+1]
        # X --> layer number, Y --> head, Z --> location
        # In our case, we dont have encoder so head is "useless"
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
        
        logger.info("--- Dataset loaded")
        self.dataloader = dataset
        self.val_dataloader = val_dataset
        logger.info("--- Using Cross Entropy Loss Function")
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
            optimizer_grouped_parameters, lr=0.1, eps=1e-06, betas=(0.9, 0.98)
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
        self.teacher.eval()

        for _ in range(self.params.n_epoch):
            logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")
            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                self.student.train()
                x, value, batch_labels = batch
                source = x[0,:]
                source_labels = value[0]
                look_up_source = batch_labels[0,:]
                base = x[-1,:]
                base_labels = value[-1]
                look_up_base = batch_labels[-1,:]

                if self.params.n_gpu > 0:
                    # x = x.to(f"cuda:0", non_blocking=True)
                    # value = value.to(f"cuda:0", non_blocking=True)
                    # Until Snellius is available, we skip this
                    pass

                self.step(
                    source_ids=source,
                    source_labels=source_labels,
                    base_ids=base,
                    base_labels=base_labels,
                    look_up_source=look_up_source,
                    look_up_base=look_up_base
                )
                iter_bar.update()
                iter_bar.set_postfix(
                    {
                        "Last_loss": f"{self.last_loss:.2f}", 
                        "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}",
                        "Last_ii_acc": f'{self.last_ii_acc:.2f}',
                        "Avg_cum_ii_acc": f"{self.total_ii_acc_epoch/self.n_iter:.2f}",
                        "Last_beh_acc": f'{self.last_beh_acc:.2f}',
                        "Avg_cum_beh_acc": f"{self.total_beh_acc_epoch/self.n_iter:.2f}"
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
        source_ids: torch.tensor, 
        source_labels: torch.tensor,
        look_up_source,
        base_ids: torch.tensor,
        base_labels: torch.tensor,
        look_up_base
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
        student_variable_names = random.choice(interchange_variable_mapping[1])

        teacher_interchanged_variables_mapping = {}
        student_interchanged_variables_mapping = {}
        # we store the interchange here.
        for i, variable in enumerate(teacher_variable_names):
            layer_index, LOC = parse_variable_name(variable)
            if layer_index in teacher_interchanged_variables_mapping:
                teacher_interchanged_variables_mapping[layer_index] += [(i, LOC)]
            else:
                teacher_interchanged_variables_mapping[layer_index] = [(i, LOC)]
        for i, variable in enumerate(student_variable_names):
            layer_index, LOC = parse_variable_name(variable)
            if layer_index in student_interchanged_variables_mapping:
                student_interchanged_variables_mapping[layer_index] += [(i, LOC)]
            else:
                student_interchanged_variables_mapping[layer_index] = [(i, LOC)]
        
        with torch.no_grad():
            # teacher forward pass normal.
            teacher_outputs = self.teacher(
                input_ids=source_ids, # source input
                labels=source_labels,
                look_up = look_up_source
            )
            # teacher forward pass normal
            dual_teacher_outputs = self.teacher(
                input_ids=base_ids, # base input
                labels=base_labels,
                look_up=look_up_base
            )
            # teacher forward pass for interchange variables.
            dual_counterfactual_activations_teacher = get_activation_at(
                self.teacher,
                input_ids=base_ids, # this is different! OBTAIN BASE ACTIVATIONS
                variable_names=teacher_variable_names,
                look_up = look_up_base
            )

            # teacher forward pass for interchanged outputs.
            counterfactual_outputs_teacher = self.teacher(
                input_ids=source_ids, # source inputs 
                labels=source_labels,
                look_up = look_up_source,
                interchanged_variables=dual_counterfactual_activations_teacher, # base activations
                variable_names=teacher_interchanged_variables_mapping
            )
            #
            counterfactual_activations_teacher = get_activation_at(
                self.teacher,
                input_ids=source_ids, # this is different! OBTAIN SOURCE ACTIVATIONS
                variable_names=teacher_variable_names,
                look_up = look_up_source
            )
            #
            dual_counterfactual_outputs_teacher = self.teacher(
                input_ids=base_ids, # base inputs 
                labels=base_labels,
                look_up = look_up_base,
                interchanged_variables=counterfactual_activations_teacher, # source activations
                variable_names=teacher_interchanged_variables_mapping
            )

        t_logits, _ = \
            teacher_outputs["logits"], teacher_outputs["hidden_states"]
        dual_t_logits, _ = \
            dual_teacher_outputs["logits"], dual_teacher_outputs["hidden_states"]
        # student forward pass normal.
        student_outputs = self.student(
            input_ids=source_ids, # source input
            t_logits=t_logits
        )
        # student forward pass normal.
        dual_student_outputs = self.student(
            input_ids=base_ids, # base input
            t_logits=dual_t_logits
        )
        # s_logits, _ = student_outputs["logits"], student_outputs["hidden_states"]
        # dual_s_logits, _ = student_outputs["logits"], student_outputs["hidden_states"]
        causal_t_logits, _ = \
            counterfactual_outputs_teacher["logits"], counterfactual_outputs_teacher["hidden_states"]
        dual_causal_t_logits, _ = \
            counterfactual_outputs_teacher["logits"], counterfactual_outputs_teacher["hidden_states"]
        
        self.last_loss = student_outputs["loss"].item()
        self.last_loss += dual_student_outputs["loss"].item()
        
        # student forward pass for interchange variables.
        dual_counterfactual_activations_student = get_activation_at(
            self.student,
            base_ids, # this is different! OBTAIN BASE ACTIVATIONS
            variable_names=student_variable_names
        )
        # counterfactual_activations_student = get_activation_at(
        #     self.student,
        #     source_ids, # this is different! OBTAIN SOURCE ACTIVATIONS
        #     variable_names=student_variable_names
        # )

        # student forward pass for interchanged outputs.
        counterfactual_outputs_student = self.student(
            input_ids=source_ids, # source input
            # intervention
            interchanged_variables=dual_counterfactual_activations_student, # base activations
            variable_names=student_interchanged_variables_mapping,
            # backward loss.
            t_logits=t_logits,
            causal_t_logits=causal_t_logits,
            # s_logits=s_logits
        )
        dual_counterfactual_outputs_student = self.student(
            input_ids=base_ids, # base input
            # intervention
            interchanged_variables=dual_counterfactual_activations_student, # source activations
            variable_names=student_interchanged_variables_mapping,
            # backward loss.
            t_logits=dual_t_logits,
            causal_t_logits=dual_causal_t_logits,
            # s_logits=dual_s_logits
        )
        self.last_loss += counterfactual_outputs_student["loss"].item()
        self.last_loss += dual_counterfactual_outputs_student["loss"].item()

        self.total_loss_epoch += self.last_loss
        self.optimize(counterfactual_outputs_student["loss"])

        self.last_ii_acc = self.ii_accuracy(teacher_variable_names, teacher_interchanged_variables_mapping,
                                            student_variable_names, student_interchanged_variables_mapping)
        self.last_beh_acc = self.beh_accuracy()
        self.total_ii_acc_epoch += self.last_ii_acc
        self.total_beh_acc_epoch += self.last_beh_acc


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
        self.n_iter += 1
        self.n_total_iter += 1
        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.student.parameters(), 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def ii_accuracy(self,
                    teacher_variable_names, teacher_interchanged_variables_mapping,
                    student_variable_names, student_interchanged_variables_mapping):
        """ Interchange intervention accuracy quantifies the extent to which the interpretable
        causal model is a proxy for the network"""
        labels = []
        predictions = []
        with torch.no_grad():
            self.student.eval()
            for batch in self.val_dataloader:
                x, _, batch_labels = batch
                source = x[0,:]
                look_up_source = batch_labels[0,:]
                base = x[-1,:]
                look_up_base = batch_labels[-1,:]
                # Run the causal model with the intervention:
                dual_counterfactual_activations_teacher = get_activation_at(
                    self.teacher,
                    base, # this is different! OBTAIN BASE ACTIVATIONS
                    variable_names=teacher_variable_names,
                    look_up=look_up_base
                )
                outputs_teacher = self.teacher(
                    input_ids=source, # source input
                    look_up=look_up_source,
                    # intervention
                    interchanged_variables=dual_counterfactual_activations_teacher, # base activations
                    variable_names=teacher_interchanged_variables_mapping
                )
                labels.append(int(outputs_teacher["logits"].argmax(dim=1)))

                # Run the neural model with the intervention:
                dual_counterfactual_activations_student = get_activation_at(
                    self.student,
                    base, # this is different! OBTAIN BASE ACTIVATIONS
                    variable_names=student_variable_names
                )
                outputs_student = self.student(
                    input_ids=source, # source input
                    # intervention
                    interchanged_variables=dual_counterfactual_activations_student, # base activations
                    variable_names=student_interchanged_variables_mapping
                )
                # Get the neural model's prediction with the intervention:
                pred = nn.Softmax(dim=1)(outputs_student['logits']).argmax(dim=1)
                predictions.append(int(pred))
            return np.sum(np.equal(predictions,labels))/len(labels)
    
    def beh_accuracy(self):
        """ Behavioral accuracy is the percentage of inputs that student agrees with teacher """
        labels = []
        predictions = []
        with torch.no_grad():
            self.student.eval()
            for batch in self.val_dataloader:
                x, _, batch_labels = batch
                source = x[0,:]
                look_up_source = batch_labels[0,:]
                base = x[-1,:]
                look_up_base = batch_labels[-1,:]
                # Run the causal model:
                outputs_teacher = self.teacher(
                    input_ids=source, # source input
                    look_up=look_up_source
                )
                labels.append(int(outputs_teacher["logits"].argmax(dim=1)))
                # Run the neural model:
                outputs_student = self.student(
                    input_ids=source # source input
                )
                # Get the neural model's prediction
                pred = nn.Softmax(dim=1)(outputs_student['logits']).argmax(dim=1)
                predictions.append(int(pred))

                # Run the causal model:
                outputs_teacher = self.teacher(
                    input_ids=base, # base input
                    look_up=look_up_base
                )
                labels.append(int(outputs_teacher["logits"].argmax(dim=1)))
                # Run the neural model:
                outputs_student = self.student(
                    input_ids=base # base input
                )
                # Get the neural model's prediction
                pred = nn.Softmax(dim=1)(outputs_student['logits']).argmax(dim=1)
                predictions.append(int(pred))

            return np.sum(np.equal(predictions,labels))/len(labels)
        
    
    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """

        self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}_loss_{self.total_loss_epoch/self.n_iter:.2f}_ii_acc_{self.total_ii_acc_epoch/self.n_iter:.2f}_beh_acc_{self.total_beh_acc_epoch/self.n_iter:.2f}.pth")

        self.epoch += 1
        self.n_iter = 0
        self.total_loss_epoch = 0
        self.total_ii_acc_epoch = 0
        self.total_beh_acc_epoch = 0

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        mdl_to_save = self.student.model[0] if hasattr(self.student.model[0], "modules") else self.student
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))

def prepare_trainer(args):

    # ARGS #
    set_seed(args)
    
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
    train_dataset, val_dataset = setup_loaders()
    logger.info("Data loader created.")

    # TRAINER #
    torch.cuda.empty_cache()
    trainer = Trainer(
        params=args,
        dataset=train_dataset,
        val_dataset=val_dataset,
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
    parser.add_argument("--n_epoch", type=int, default=10, help="Number of pass on the whole dataset.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (for each process).")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
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


