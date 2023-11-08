import torch
import torchvision.models as models
from torch import nn
from utils import _get_value

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
        student_output = {}
        student_output["hidden_states"]=[]
        # Interchange intervention
        hidden_state = input_ids.unsqueeze(0)
        layers = [self.model[0].conv1, self.model[0].maxpool, self.model[0].layer1, self.model[0].layer2, self.model[0].layer3, self.model[0].layer4]
        for i, layer_module in enumerate(layers):
            layer_outputs = layer_module(
                hidden_state
            )
            hidden_state = layer_outputs
            if variable_names != None and i in variable_names:
                assert interchanged_variables != None
                for interchanged_variable in variable_names[i]:
                    interchanged_activations = interchanged_variables[interchanged_variable[0]]
                    start_index = interchanged_variable[2].start
                    stop_index = interchanged_variable[2].stop
                    hidden_state[...,start_index:stop_index] = interchanged_activations
                    #Actually return the interchanged hidden states!!!
                    student_output["hidden_states"].append(hidden_state)
        
        # return "normal" activations if not interchange
        x = self.model[0].conv1(input_ids.unsqueeze(0))
        student_output["hidden_states"].append(x) if not interchanged_variables else student_output
        x = self.model[0].maxpool(x)
        student_output["hidden_states"].append(x) if not interchanged_variables else student_output
        x = self.model[0].layer1(x)
        student_output["hidden_states"].append(x) if not interchanged_variables else student_output
        x = self.model[0].layer2(x)
        student_output["hidden_states"].append(x) if not interchanged_variables else student_output
        x = self.model[0].layer3(x)
        student_output["hidden_states"].append(x) if not interchanged_variables else student_output
        x = self.model[0].layer4(x)
        student_output["hidden_states"].append(x) if not interchanged_variables else student_output
        x = self.model[0].avgpool(x)
        x = torch.flatten(x, 1)

        student_output["logits"] = self.model[0].fc(x)
        
        if labels is not None:
            tensor_labels = torch.zeros((1,10))
            tensor_labels[0,labels.item()]=1
            student_output["loss"]  = self.loss(student_output["logits"] , tensor_labels)
        
        # Double backward update
        if causal_t_logits is None:
            if t_logits is not None:
                assert t_hidden_states is not None
                s_logits, _ = student_output["logits"], student_output["hidden_states"]
                loss = self.loss(s_logits, t_logits)
                student_output["loss"] += loss

        return student_output