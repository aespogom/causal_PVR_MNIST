import numpy as np
import torch


def deserialize_variable_name(variable_name):
    deserialized_variables = []
    variable_list = variable_name.split("$")
    if "[" in variable_list[1]:
        left_l = int(variable_list[1].split(":")[1].strip("["))
        right_l = int(variable_list[1].split(":")[2].strip("]"))
    else:
        left_l = int(variable_list[1].split(":")[-1])
        right_l = int(variable_list[1].split(":")[-1])+1
    if "[" in variable_list[2]:
        left_h = int(variable_list[2].split(":")[1].strip("["))
        right_h = int(variable_list[2].split(":")[2].strip("]"))
    else:
        left_h = int(variable_list[2].split(":")[-1])
        right_h = int(variable_list[2].split(":")[-1])+1

    left_d = int(variable_list[3].split(":")[0].strip("["))
    right_d = int(variable_list[3].split(":")[1].strip("]"))
    
    for i in range(left_l, right_l):
        for j in range(left_h, right_h):
            deserialized_variable = f"$L:{i}$H:{j}$[{left_d}:{right_d}]"
            deserialized_variables += [deserialized_variable]
    return deserialized_variables


def parse_variable_name(variable_name, model_config=None):
    if model_config == None:
        variable_list = variable_name.split("$")
        layer_number = int(variable_list[1].split(":")[-1])
        head_number = int(variable_list[2].split(":")[-1])
        LOC_left = int(variable_list[3].split(":")[0].strip("["))
        LOC_right = int(variable_list[3].split(":")[1].strip("]"))
        LOC = np.s_[LOC_left:LOC_right]
        return layer_number, head_number, LOC
    else:
        # to be supported.
        pass
    
def get_activation_at(
    model, input_ids,
    variable_names
):
    outputs = model(
        input_ids
    )
    activations = []
    for variable in variable_names:
        layer_index, head_index, LOC = parse_variable_name(
            variable_name=variable
        )
        head_hidden_states = outputs["hidden_states"][layer_index][
            :,(head_index):((head_index+1))
        ]
        head_slice = head_hidden_states[:,LOC]
        activations += [head_slice]
    return activations
