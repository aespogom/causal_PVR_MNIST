## TEACHER MODEL
import torch
from torch import nn
from utils import _get_value

class Oracle(nn.Module):
    """
    Our target causal model will abstract away from the details of how to identify
        the handwritten digit in an image, focusing just on the reasoning about pointers.
    Formally, we define a causal model C = (V, PA, Val, F) that computes the label for each of the four
        MNIST images using an oracle OMNIST with a look-up table to select the correct label based on the pointer.
    The parents are defined such that PA_Iw = ∅ and PA_Yw = {Iw} for all w ∈ {topright, topleft, bottomright, bottomleft}, and PA_O = {Y_topleft, Y_topright, Y_bottomleft, Y_bottomright}.
    The structural equations are:
        FY_topleft (I_topleft) = OMNIST(I_topleft)
        FY_topright (I_topright) = OMNIST(I_topright)
        FY_bottomleft (I_bottomleft) = OMNIST(I_bottomleft)
        FY_bottomright (I_bottomright) = OMNIST(I_bottomright)

        FO(Y_topleft, Y_topright, Y_bottomleft, Y_bottomright) = yTR  if yTL ∈ {0, 1, 2, 3} // yBL if yTL ∈ {4, 5, 6} // yBR if yTL ∈ {7, 8, 9}
    """
    def __init__(self):
        super().__init__()
        # Oracle lookup
        self.model = self.oracle_look_up_table
        self.loss = nn.MSELoss()

    def oracle_look_up_table(self, index, look_up):
        return look_up[index]

    def forward(
        self,
        input_ids,
        labels=None,
        look_up = None,
        # for interchange.
        interchanged_variables=None, 
        variable_names=None,
        interchanged_activations=None
    ):
        """
        Inputs:
            input_ids: inputs
            labels: real labels
            interchanged_variables: alignment, 
            variable_names: mapping,
            interchanged_activations: values to interchange
        """
        tl = input_ids[:, :40, :40].unsqueeze(dim=0)
        tr = input_ids[:, :40, 40:].unsqueeze(dim=0)
        bl = input_ids[:, 40:, :40].unsqueeze(dim=0)
        br = input_ids[:, 40:, 40:].unsqueeze(dim=0)
        list_inputs = [tl, tr, bl, br]

        teacher_ouputs = {}
        teacher_ouputs["hidden_states"]=[]
        # we perform the interchange intervention
        for i, x in enumerate(list_inputs):
            x = self.oracle_look_up_table(i, look_up)
            # we need to interchange!
            if variable_names != None and i in variable_names:
                assert interchanged_variables != None
                for interchanged_variable in variable_names[i]:
                    interchanged_activations = interchanged_variables[interchanged_variable[0]]
                    x = interchanged_activations
            
            teacher_ouputs["hidden_states"].append(x)
        
        FO = _get_value([
            teacher_ouputs["hidden_states"][0], # FY_topleft
            teacher_ouputs["hidden_states"][1], # FY_topright
            teacher_ouputs["hidden_states"][2], # FY_bottomleft
            teacher_ouputs["hidden_states"][3]  # FY_bottomright
        ])
        tensor_preds = torch.zeros((1,10))
        tensor_preds[0,FO.item()]=1
        teacher_ouputs["outputs"]=tensor_preds

        return teacher_ouputs
