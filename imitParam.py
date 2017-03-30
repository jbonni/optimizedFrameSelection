import argparse



"""
  Parser containing all parameters for the imitation learner.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--input_shape',
                    type=int,
                    default=[None, 784],
                    help='Input shape for the network')

parser.add_argument('--filename',
                    type = str,
                    default='mnist.hdf5',
                    help='filename for the training. The file must be in the \
                         working directory')

parser.add_argument('--output_shape',
                    type=float,
                    default=[None, 10],
                    help='Output shape for the network.')

parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='Initial learning rate.')

parser.add_argument('--max_steps',
                    type=int,
                    default=35000,
                    help='Number of steps to run trainer.')

parser.add_argument('--input_shape',
                    type=int,
                    default=[-1,28,28,1],
                    help='Shape of the input of the neural network. Should be\
                          the same shape as the input data')

parser.add_argument('--Conv_param',
                    type=int,
                    default=[[32,8,8,4],\
                             [64,4,4,2],\
                             [64,3,3,2]],
                    help='Parameters of convolutional layers: \
                         filters,kernel, kernel, stride.\
                         Simpli append parameters to the list to add layers')
parser.add_argument('--fully_connected_layer',
                    type=int,
                    default=[[512]],
                    help='Number of units in each FCL.\
                          Add elements to the list to configure the fully\
                          connected network.')

parser.add_argument('--batch_size',
                    type=int,
                    default=64,
                    help='Batch size. \
                          Must divide evenly into the dataset sizes.')

parser.add_argument('--input_data_dir',
                    type=str,
                    default='/tmp/tensorflow/mnist/input_data',
                    help='Directory to put the input data.')

parser.add_argument('--log_dir',
                    type=str,
                    default='/tmp/tensorflow/mnist/logs/fully_connected_feed',
                    help='Directory to put the log data.')

imitParam = parser.parse_args()