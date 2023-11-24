import torch
import torchvision.models as models
from torch import nn
from counterfactual_utils import interchange_hook
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
        self.model = resnet18
        self.loss = nn.CrossEntropyLoss()
        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")

    def forward(self,
                input_ids,
                # for interchange.
                interchanged_variables=None, 
                variable_names=None,
                interchanged_activations=None,
                # # losses
                t_outputs=None,
                causal_t_outputs=None,
                s_outputs=None
                ):
        student_output = {}
        student_output["hidden_states"]=[]
        # Interchange intervention
        x = input_ids.unsqueeze(0)
        layers = [self.model.conv1, self.model.maxpool, self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        hooks = []
        for i, layer_module in enumerate(layers):
            if variable_names != None and i in variable_names:
                assert interchanged_variables != None
                for interchanged_variable in variable_names[i]:
                    interchanged_activations = interchanged_variables[interchanged_variable[0]]
                    #https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/#method-3-attach-a-hook AND interchange_with_activation_at()
                    hook = layer_module.register_forward_hook(interchange_hook(interchanged_variable, interchanged_activations))
                    hooks.append(hook)
            x = layer_module(
                x
            )
            student_output["hidden_states"].append(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        student_output["outputs"] = self.model.fc(x)

        ## Origin code uses softmax before loss because they use KL divergence loss
        ## But this is not the case for crossentropy
        # student_output["outputs"] = nn.Softmax(dim=1)(self.model[0].fc(x))
        
        # This part is only teacher
        # # if labels is not None:
        # #     tensor_labels = torch.zeros((1,10))
        # #     tensor_labels[0,labels.item()]=1
        # #     student_output["loss"]  = self.loss(student_output["outputs"] , tensor_labels)
        
        # IIT Objective
        # For each intermediate variable Yw ∈ {YTL, YTR, YBL, YBR}, we introduce an IIT
        #    objective that optimizes for N implementing Cw the
        #    submodel of C where the three intermediate variables
        #    that aren’t Yw are marginalized out:
        #     sum[ CE(Cw intinv, N intinv)]
        if causal_t_outputs is None:
            # if it is None, it is simply a forward for getting hidden states!
            if t_outputs is not None:
                s_outputs, _ = student_output["outputs"], student_output["hidden_states"]
                loss = self.loss(s_outputs, t_outputs)
                student_output["loss"] = loss
        else:
            # causal loss.
            causal_s_outputs, _ = \
                student_output["outputs"], student_output["hidden_states"]
            loss = self.loss(causal_s_outputs, causal_t_outputs)
            student_output["loss"] = loss

            # measure the efficacy of the interchange.
            teacher_interchange_efficacy = (
                self.ce_loss_fct(
                    nn.functional.log_softmax(causal_t_outputs, dim=-1),
                    nn.functional.softmax(t_outputs, dim=-1),
                )
            )

            student_interchange_efficacy = (
                self.ce_loss_fct(
                    nn.functional.log_softmax(causal_s_outputs, dim=-1),
                    nn.functional.softmax(s_outputs, dim=-1),
                )
            )
            student_output["teacher_interchange_efficacy"] = teacher_interchange_efficacy
            student_output["student_interchange_efficacy"] = student_interchange_efficacy
        
        for h in hooks:
            h.remove()

        return student_output