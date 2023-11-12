import numpy as np

def deserialize_variable_name(variable_name):
    deserialized_variables = []
    variable_list = variable_name.split("$")
    if "[" in variable_list[1]:
        left_l = int(variable_list[1].split(":")[1].strip("["))
        right_l = int(variable_list[1].split(":")[2].strip("]"))
    else:
        left_l = int(variable_list[1].split(":")[-1])
        right_l = int(variable_list[1].split(":")[-1])+1

    left_d = variable_list[2].split(",")[0].strip("[")
    right_d = variable_list[2].split(",")[1].strip("]")
    
    for i in range(left_l, right_l):
        deserialized_variable = f"$L:{i}$[{left_d},{right_d}]"
        deserialized_variables += [deserialized_variable]
    return deserialized_variables


def parse_variable_name(variable_name, model_config=None):
    """
    LOC locality of convolution
    """
    if model_config == None:
        variable_list = variable_name.split("$")
        layer_number = int(variable_list[1].split(":")[-1])
        LOC_left = variable_list[2].split(",")[0].strip("[")
        LOC_right = variable_list[2].split(",")[1].strip("]")
        LOC = np.s_[LOC_left:LOC_right]
        return layer_number, LOC
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
        layer_index, LOC = parse_variable_name(
            variable_name=variable
        )
        # TODO NOT REUSABLE AT ALL, NEED TO GENERALIZE
        if LOC.start == LOC.stop and LOC.start==":":
            head_slice = outputs["hidden_states"][layer_index]
        else:
            if LOC.start == ':10':
                if LOC.stop==':10':
                    head_slice = outputs["hidden_states"][layer_index][...,:10,:10]
                else:
                    head_slice = outputs["hidden_states"][layer_index][...,:10,10:]
            else:
                if LOC.stop==':10':
                    head_slice = outputs["hidden_states"][layer_index][...,10:,:10]
                else:
                    head_slice = outputs["hidden_states"][layer_index][...,10:,10:]
            
        activations += [head_slice]
    return activations

def interchange_hook(interchanged_variable, interchanged_activations):
    # the hook signature
    def hook(model, input, output):
        # interchange inplace.
        # TODO NOT REUSABLE AT ALL, NEED TO GENERALIZE
        if interchanged_variable[1].start == ':10':
            if interchanged_variable[1].stop==':10':
                output[...,:10,:10] = interchanged_activations
            else:
                output[...,:10,10:] = interchanged_activations
        elif interchanged_variable[1].start == '10:':
            if interchanged_variable[1].stop==':10':
                output[...,10:,:10] = interchanged_activations
            else:
                output[...,10:,10:] = interchanged_activations
        else:
            output = interchanged_activations
    return hook