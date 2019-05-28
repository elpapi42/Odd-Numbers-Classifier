import tensorflow as tf
import numpy as np

class Unit(object):
    """
    Single Compute Unit representing a Neuron, Basic block of neural networks

    If you want to use a single unit, use an instance of Classes.DenseLayer with units=1

    Args:
        input (tf.Tensor): Input tensor of the Unit
        name (str): Name for group ops in the Unit

    Attributes:
        input (tf.Tensor): Input tensor of the Unit
        name (str): Name for group ops in the Unit
        weight (tf.Variable): Stores weights adapted to input.shape
        bias (tf.Variable): Store bias adapted to input rank - 1
        output (tf.Tensor): Output tensor of the Unit

    Raises:
        ValueError: Input must have shape=[batch_size, input_size]

    """

    def __init__(self, input=None, name="node"):

        if(input.shape.rank == 2):
            with tf.name_scope(name):
                self.name = name
            
                self.input = tf.to_float(input, "input/unit_input")
                self.weight = tf.get_variable(name + "/weight", [self.input.shape[1], 1], tf.float32, tf.initializers.random_uniform(0.0, 1.0))
                self.bias = tf.get_variable(name + "/bias", [], tf.float32, tf.initializers.random_uniform(0.0, 1.0))

                self.weighted_sum = tf.matmul(self.input, self.weight, name="weighted_sum")
                self.bias_add = tf.add(self.weighted_sum, self.bias, "bias_add")
                self.output = tf.nn.relu(self.bias_add, "relu_output")
        else:
            raise ValueError("Input must have shape=[batch_size, input_size]")
            
    def get_input(self):
        """Returns the input of the Unit"""
        return self.input

    def get_output(self):
        """Returns the output of the Unit"""
        return self.output

    def get_weight(self):
        """Returns the weight of the Unit"""
        return self.weight

    def get_bias(self):
        """Returns the bias of the Unit"""
        return self.bias

    def get_name(self):
        """Returns name of the Unit"""
        return self.name


