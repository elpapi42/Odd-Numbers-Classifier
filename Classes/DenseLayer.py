from Classes.Unit import Unit as Unit
import tensorflow as tf
import numpy as np

class DenseLayer(object):
    """
    Layer of Units that are fully connected to the input and output, mimics tf.Layers.Dense
    
    Args:
        input (tf.Tensor): Input tensor of the Layer
        units (int): Number of Units generated in the Layer
        name (str): Name for group Layer Units

    Attributes:
        input (tf.Tensor): Input tensor of the Layer
        name (str): Name for group Units in the Layer
        units_list (tf.Tensor): List of tensors from the output of every Unit in the Layer
        output (tf.Tensor): Output of the Layer, Stack the output of every Unit in the Layer

    Raises:
        ValueError: units must be greater than zero
        ValueError: Input must have rank <=2
        
    """

    def __init__(self, input=None, units=1, name="layer"):

        if(units >= 1):
            self.name = name
            self.units = units

            with tf.name_scope(self.name + "/input/"):
                if(input.shape.rank == 2):
                    self.input = tf.to_float(input, "input")
                elif(input.shape.rank == 0):
                    self.input = tf.to_float(tf.reshape(input, [1, 1], "reshaped_input_from_scalar"), "input")
                elif(input.shape.rank == 1):
                    input_shape = tf.shape(input, name="input_shape")
                    self.input = tf.to_float(tf.reshape(input, [input_shape[0], 1], "reshaped_input_from_vector"), "input")
                else:
                    raise ValueError("Input must have rank <=2")

            self.units_list = list()
            
            for i_unit in range(self.units):
                unit = Unit(self.input, name + "/unit_" + str(i_unit)).get_output()
                self.units_list.append(unit)

            with tf.name_scope(self.name + "/output/"):
                units_output_shape = tf.shape(self.units_list[0], name="units_output_shape")
                self.output = tf.reshape(tf.stack(self.units_list, 1, name="stack"), [units_output_shape[0], self.units], "output")
        else:
            raise ValueError("units must be greater than zero")
  
    def get_output(self):
        """Returns output of the Layer"""
        return self.output

    def get_units_list(self):
        """Returns list of output tesnors from every Unit in the Layer"""
        return self.units_list

    def get_input(self):
        """Returns input of the Layer"""
        return self.input

    def get_name(self):
        """Returns name of the Layer"""
        return self.name


