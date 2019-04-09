import glob
from redbaron import RedBaron


help_dict = {
    '--model-dir': 'Location to write checkpoints and summaries to.  Must be a GCS URI when using Cloud TPU.',
    '--max-steps': 'The total number of steps to train the model.',
    '--use-tpu': 'Whether to use TPU.',
    '--tpu': 'The name or GRPC URL of the TPU node.  Leave it as `None` when training on AI Platform.',
    '--train-batch-size': 'The training batch size.  The training batch is divided evenly across the TPU cores.',
    '--save-checkpoints-steps': 'The number of training steps before saving each checkpoint.',
    '--sequence-length': 'The sequence length for an LSTM model.',
    '--gr-weight': 'The weight used in the gradient reversal layer.',
    '--lambda': 'The trade-off between label_prediction_loss and domain_classification_loss.'
}

filenames = glob.glob('../**/trainer*.py')

for filename in filenames:
    with open(filename, 'r') as f:
        red = RedBaron(f.read())

    # the `if __name__ == '__main__':` block
    ifelseblock = red[-1]
    nodes = ifelseblock.value[0]

    for node in nodes:
        # looking for those parser.add_argument calls
        if node.type != 'atomtrailers':
            continue

        # reference on the node structure: https://redbaron.readthedocs.io/en/latest/nodes_reference.html#atomtrailersnode
        if node.value[0].name.value != 'parser' or node.value[1].name.value != 'add_argument':
            continue

        # args passed into the add_argument call
        args = node.value[2].value

        # get the arg name
        assert args[0].target is None
        arg_name = args[0].value.value.replace("'", '')

        assert arg_name.startswith('--')

        # check if a `help` argument has already been passed in, and if not, add it.
        for arg in args[1:]:
            if arg.target.value == 'help':
                break

        else:
            # create a CallArgumentNode for the `help` keyward argument
            arg = args[-1].copy()
            arg.target.value = 'help'
            node.value[2].value.append(arg)

        if arg_name in help_dict:
            help_string = "'" + help_dict.get(arg_name, '') + "'"
        else:
            print('>>> {} does not have a help string.'.format(arg_name))
            help_string = "''"

        arg.value.value = help_string

    with open(filename, 'w') as out:
        out.write(red.dumps())
