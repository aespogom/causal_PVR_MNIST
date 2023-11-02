import argparse
import json
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torchvision.models as models
from torchvision import datasets
from torchvision import transforms

import math
import os
import time
import shutil

from tqdm import tqdm
from counterfactual_utils import deserialize_variable_name, parse_variable_name, get_activation_at
import multiprocessing
from utils import logger, set_seed
from transformers import get_linear_schedule_with_warmup

AVAIL_GPUS = min(1, torch.cuda.device_count())
AVAIL_CPUS = multiprocessing.cpu_count()

## DATASET
TRAIN_SET = datasets.MNIST(root='data', train=True, download=True)
TEST_SET = datasets.MNIST(root='data', train=False, download=True)

HOLDOUT_CLASSES = {0: [],  # top_left
                   1: [1, 2, 3],  # top_right
                   2: [4, 5, 6],  # bottom_left
                   3: [7, 8, 9, 0]}  # bottom_right

TRANSLATION_FACTOR = (40 - 28) / 2 / 40
TRANSFORM = transforms.Compose([
    transforms.CenterCrop([40, 40]),  # basically pads with zeros
    transforms.RandomAffine(0, translate=[TRANSLATION_FACTOR, TRANSLATION_FACTOR]),
    transforms.ToTensor()
])


def _get_value(labels):
    """
    get the value based on the pointer

    Args:
        labels: a tensor of length 4. where labels[0] is the pointer

    Returns:
        The value
    """

    pointer = labels[0]
    if 0 <= pointer <= 3:
        value = labels[1]
    elif 4 <= pointer <= 6:
        value = labels[2]
    else:
        value = labels[3]

    return value

class BlockStylePVR(Dataset):
    def __init__(self,
                 train: bool,
                 mode: str = "iid",
                 size: int = None):
        """

        Args:
            train: whether to use MNIST train set, else use the test set.
            mode: "holdout" or "adversarial" or "iid".
            size: dataset size
        """

        self.ds = TRAIN_SET if train else TEST_SET
        #size train is 60.000
        if size is not None and size > len(self.ds) // 4:
            raise ValueError(f"Requested dataset size is too big. Can be up too {len(self.ds) // 4}.")

        # use maximum size if size is null
        self.pvr_ds_size = len(self.ds) // 4 if size is None else int(size)

        # the labels (0-9) of the images in the dataset
        ds_labels = []
        for idx, (c, y) in enumerate(self.ds): #idx index, _ image PIL, y label
            ds_labels.append(y)
        ds_labels = torch.tensor(ds_labels)

        # dtype must be torch.long, otherwise it will crash during training
        self.idxs = torch.zeros([self.pvr_ds_size, 4], dtype=torch.long)
        self.labels = torch.zeros([self.pvr_ds_size, 4], dtype=torch.long)  # 0-9

        if mode == 'iid':
            # the labels (0-9) of the 4 digits in each sample
            self.labels = ds_labels.reshape([-1, 4])  # group into 4's
            self.labels = self.labels[:self.pvr_ds_size]  # trim to requested size

            # the original idx of the 4 digits in each sample
            self.idxs = torch.arange(len(self.ds)).reshape([-1, 4])  # group into 4's
            self.idxs = self.idxs[:self.pvr_ds_size]  # trim to requested size

        elif mode == "holdout":
            # sample from the ds excluding the labels that are held out
            for i, holdout_class in HOLDOUT_CLASSES.items():
                probs = torch.ones(len(self.ds))
                for label in holdout_class:
                    probs[ds_labels == label] = 0
                curr_idxs = torch.multinomial(probs, self.pvr_ds_size)
                self.idxs[:, i] = curr_idxs
                self.labels[:, i] = ds_labels[curr_idxs]

        elif mode == "adversarial":
            # sample from the ds only where the labels are held out
            for i, holdout_class in HOLDOUT_CLASSES.items():
                probs = torch.ones(len(self.ds)) if i == 0 else torch.zeros(len(self.ds))
                for label in holdout_class:
                    probs[ds_labels == label] = 1
                curr_idxs = torch.multinomial(probs, self.pvr_ds_size)
                self.idxs[:, i] = curr_idxs
                self.labels[:, i] = ds_labels[curr_idxs]

        else:
            raise ValueError("Unknown dataset mode.")

    def __getitem__(self, idx):
        labels = self.labels[idx]
        idxs = self.idxs[idx]

        # for each label, get the matching image from the ds using the idx
        # transform it (fit into 40x40 and translate randomly)
        # and then put in the appropriate location in the result image
        x = torch.zeros([1, 80, 80])
        x[0, :40, :40] = TRANSFORM(self.ds[idxs[0]][0])
        x[0, :40, 40:] = TRANSFORM(self.ds[idxs[1]][0])
        x[0, 40:, :40] = TRANSFORM(self.ds[idxs[2]][0])
        x[0, 40:, 40:] = TRANSFORM(self.ds[idxs[3]][0])

        # calculate the value based on the pointer
        value = _get_value(labels)

        return x, value

    def __len__(self):
        return self.pvr_ds_size

def setup_loaders():
    torch.manual_seed(42)

    num_workers = (4 * AVAIL_GPUS) if (AVAIL_GPUS > 0) else AVAIL_CPUS

    shuffle = False
    # Aumentar size para dataset completo
    train_ds = BlockStylePVR(train=True, size=10)
    train_loader = DataLoader(train_ds, batch_size=3,
                              pin_memory=True,
                              num_workers=num_workers,
                              shuffle=shuffle)

    val_ds = BlockStylePVR(train=False, size=10)
    val_loader = DataLoader(val_ds, batch_size=2,
                            pin_memory=True,
                            num_workers=num_workers)

    return train_loader, val_loader

## STUDENT MODEL
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        # get original model
        resnet18 = models.resnet18()

        # modify input channels
        input_layer = resnet18.conv1
        resnet18.conv1 = nn.Conv2d(1,
                                   input_layer.out_channels,
                                   kernel_size=input_layer.kernel_size,
                                   stride=input_layer.stride,
                                   padding=input_layer.padding,
                                   bias=input_layer.bias)

        # modify last layer to match output classes
        num_features = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_features, 10)

        self.model = nn.ModuleList([resnet18])
        self.loss = nn.CrossEntropyLoss()

    def forward(self,
                input_ids,
                labels=None,
                # for interchange.
                interchanged_variables=None, 
                variable_names=None,
                interchanged_activations=None,
                # # losses
                t_logits=None,
                t_hidden_states=None,
                causal_t_logits=None,
                causal_t_hidden_states=None,
                s_logits=None,
                s_hidden_states=None,
                lm_labels=None
                ):
        
        # Interchange intervention
        hidden_state = input_ids
        for i, layer_module in enumerate(self.model):
            all_hidden_states = () + (hidden_state,)
            layer_outputs = layer_module(
                input_ids
            )
            hidden_state = layer_outputs[-1]
            
            # we need to interchange!
            if variable_names != None and i in variable_names:
                assert interchanged_variables != None
                for interchanged_variable in variable_names[i]:
                    interchanged_activations = interchanged_variables[interchanged_variable[0]]
                    start_index = interchanged_variable[1] + interchanged_variable[2].start
                    stop_index = start_index + interchanged_variable[2].stop
                    hidden_state[...,start_index:stop_index] = interchanged_activations

            layer_outputs[-1] = hidden_state
        
        student_output = {}
        student_output["hidden_states"]=[]
        x = self.model[0].conv1(input_ids)
        student_output["hidden_states"].append(x)
        x = self.model[0].layer1(x)
        student_output["hidden_states"].append(x)
        x = self.model[0].layer2(x)
        student_output["hidden_states"].append(x)
        x = self.model[0].layer3(x)
        student_output["hidden_states"].append(x)
        x = self.model[0].layer4(x)
        student_output["hidden_states"].append(x)
        x = self.model[0].avgpool(x).squeeze()
        
        student_output["logits"] = self.model[0].fc(x)
        
        if labels is not None:
            label_tensor = torch.zeros(3,10)
            label_tensor[:,_get_value(labels)]=1
            student_output["loss"]  = self.loss(student_output["logits"] , label_tensor)
        
        # if causal_t_logits is None:
        #     if t_logits is not None:

        return student_output

## TEACHER MODEL
class Oracle(nn.Module):
    def __init__(self):
        super().__init__()
        # 80 xq el input es 80x80 tensores, 10 xq el output es clasificacion de 10 labels 
        self.model = nn.ModuleList([nn.Linear(80, 10)])
        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        labels=None,
        # for interchange.
        interchanged_variables=None, 
        variable_names=None,
        interchanged_activations=None
    ):
        r"""
        Formally, we define a causal model CPVR =
        (V, PA, Val, F) that computes the label for each of the four
        MNIST images using an oracle OMNIST with a look-up table
        to select the correct label based on the pointer.

        Inputs:
            input_ids: inputs
            labels: real labels
            interchanged_variables: alignment, 
            variable_names: mapping,
            interchanged_activations: values to interchange
        """
        # we perform the interchange intervention
        hidden_states = input_ids
        for i, layer_module in enumerate(self.model):
            layer_outputs = layer_module(
                hidden_states
            )
            hidden_states = layer_outputs[-1]
            # we need to interchange!
            if variable_names != None and i in variable_names:
                assert interchanged_variables != None
                for interchanged_variable in variable_names[i]:
                    interchanged_activations = interchanged_variables[interchanged_variable[0]]
                    start_index = interchanged_variable[1] + interchanged_variable[2].start
                    stop_index = start_index + interchanged_variable[2].stop
                    hidden_states[...,start_index:stop_index] = interchanged_activations
            layer_outputs[-1] = hidden_states
        # Necesitamos el moduleLayer para el iit pero realmente solo hay una layer
        pred_scores = self.model[0](input_ids)
        teacher_ouputs = {}        
        teacher_ouputs["logits"]=pred_scores
        teacher_ouputs["hidden_states"]=layer_outputs[-1]
        if labels is not None:
            label_tensor = torch.zeros(3,10)
            label_tensor[:,_get_value(labels)]=1
            teacher_ouputs["loss"] = self.loss(teacher_ouputs["logits"].squeeze() , label_tensor.long())

        return teacher_ouputs

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

    student_model = ResNet18()
    # student = student_model.to(f"cuda:0", non_blocking=True)
    logger.info("Student loaded.")
    teacher_model = Oracle()
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


