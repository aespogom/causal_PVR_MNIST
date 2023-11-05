## TEACHER MODEL
import torch
from torch import nn

from utils import _get_value

class Oracle(nn.Module):
    def __init__(self):
        super().__init__()
        # The nn.Flaten is responsible for transforming the data from multidimensional to one dimension only.
        # 6400 xq el input es 80x80 tensores, 10 xq el output es clasificacion de 10 labels 
        self.model = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(80*80, 10)
        ])
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

        teacher_ouputs = {}
        teacher_ouputs["hidden_states"]=[]
        x = self.model[0](input_ids) #flatten
        teacher_ouputs["hidden_states"].append(x)
        pred_scores = self.model[1](x) #linear    
        
        teacher_ouputs["logits"]=pred_scores

        if labels is not None:
            label_tensor = torch.zeros(3,10)
            label_tensor[:,_get_value(labels)]=1
            teacher_ouputs["loss"] = self.loss(teacher_ouputs["logits"].squeeze() , label_tensor)

        return teacher_ouputs
