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