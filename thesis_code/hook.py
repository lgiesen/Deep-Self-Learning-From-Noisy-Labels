# define the hook function with the 
# intermediate_output as the features
def hook(module, input, output):
    global intermediate_output
    intermediate_output = output.detach()

def reset_intermediate_output():
    global intermediate_output
    intermediate_output = None

# register the hook to the layer 
# before the FC layer (AdaptiveAvgPool2d)
hook = model.avgpool.register_forward_hook(hook)