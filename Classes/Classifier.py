import tensorflow as tf
import numpy as np
import math
import time as dt
from Classes.DenseLayer import DenseLayer as dl

class Classifier(object):
    """
    General Classifier of data, takes the input values and classify them in a class array

    Args:
        input_data (list): train/test data for the model, must have shape [batch_size, input_size]
        output_data (list): train/test data for the model, must have shape [batch_size, output_size]
        model_shape (list): list of ints, the lenght of the list represent the number of layers in the model, 
                            and the value of each index represents the number of units in each layer
        mini_batch_size (int): the number of inputs that will be taken from the input data and feed into the model each train iteration
        learning_rate (float): learning rate of the model

    Attributes:
        mini_batch_size (int): the number of inputs that will be taken from the input data and feed into the model each train iteration
        input_size (int): size of each element in input_data
        output_size (int): size of each element in output_data
        batch_size (int): lenght of input and output data, the total number of traning/test samples provided
        use_test_data (tf.Operation): set the test data as input to the model, used for keep track of the model performance
        use_train_data (tf.Operation): set the train data as input to the model, used for train the model
        input (tf.Tensor): input of the model
        output (tf.Tensor): output of the model
        loss (tf.Tensor): loss of the model in the last inference
        train_op (tf.Operation): operation used for recalculate the weights and biases of the model
        sess (tf.Session): session used for train/validate/inference
        summary (tf.Operation): records all the summarized data in the model when called from sess.run()
        sum_writer (tf.summary.FileWriter): write the recorded data to disk
        epochs (int): number of training cycles perform to this instance of the model

    Raises:
        TypeError: input_data must have shape [batch_size, input_size]
        TypeError: output_data must have shape [batch_size, output_size]
        ValueError: batch_size must be equal in input_data and output_data
        TypeError: model_shape cant be a scalar

        
    """

    def __init__(self, input_data, output_data, model_shape=[1], mini_batch_size=1, learning_rate=0.1):
        
        if(len(np.shape(input_data)) == 2 and len(np.shape(output_data)) == 2 and len(input_data) == len(output_data) and np.shape(model_shape) != ()):

            self.input_size = len(input_data[0])
            self.output_size = len(output_data[0])
            self.batch_size = len(input_data)
            self.mini_batch_size = mini_batch_size

            #pack input and output data in a single data list, the suffle and divide in test data and train data
            data = np.concatenate((input_data, output_data), 1)
            np.random.shuffle(data)
            train_data, test_data = np.split(data, [int(0.7 * len(data))])

            with tf.name_scope("Parameters/"):

                with tf.name_scope("Parameters/Dataset/"):

                    test_dataset = tf.data.Dataset.from_tensor_slices(test_data).repeat().batch(len(test_data))
                    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).repeat().batch(mini_batch_size)
                    
                    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes, "data_iterator/iterator")
                    next_item = iterator.get_next("next_item")

                    #Operations for switch train/test datasets
                    self.use_test_data = iterator.make_initializer(test_dataset)
                    self.use_train_data = iterator.make_initializer(train_dataset)

                #slice the next element that contains packed the input and output data
                next_input = tf.slice(next_item, [0, 0], [-1, self.input_size], "input_slice")
                next_output = tf.slice(next_item, [0, self.input_size], [-1, self.output_size], "output_slice")

                #model input, if no inference is done, pick the input from the actual dataset
                self.input = tf.placeholder_with_default(next_input, [None, self.input_size], "model_input")

            layer_list = list()
            last_output = self.input

            #builds the layers of the model
            for layer_index in range(len(model_shape)):
                layer = dl(last_output, model_shape[layer_index], "Layers/layer_" + str(layer_index))
                last_output = layer.get_output()
                layer_list.append(layer)
                
            #output layer with fixed lenght based on output_size
            output_layer = dl(last_output, self.output_size, "Layers/output_layer")
                
            with tf.name_scope("Output/"):
                self.output = tf.nn.softmax(output_layer.get_output(), name="Softmax")

            with tf.name_scope("Correction/"):
                self.loss = tf.losses.mean_squared_error(next_output, self.output)
                tf.summary.scalar("loss", self.loss)
                self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.summary = tf.summary.merge_all()
            self.sum_writer = tf.summary.FileWriter("./RecordSummary/" + 
                                                    "lr=" + str(learning_rate) +
                                                    ".-mbs=" + str(self.mini_batch_size) +
                                                    ".-bs=" + str(self.batch_size) +
                                                    ".-ms=" + str(model_shape) +
                                                    ".-date=" + str(dt.strftime("%d.%m.%Y.%H.%M.%S")))
            self.sum_writer.add_graph(self.sess.graph)
            self.epochs = 0
        else:
            if(len(np.shape(input_data)) != 2):
                raise TypeError("input_data must have shape [batch_size, input_size]")
            elif(len(np.shape(output_data)) != 2):
                raise TypeError("output_data must have shape [batch_size, output_size]")
            elif(len(input_data) != len(output_data)):
                raise ValueError("batch_size must be equal in input_data and output_data")
            elif(np.shape(model_shape) == ()):
                raise TypeError("model_shape cant be a scalar")
            
    def train(self, epochs=1):
        """
        Train the model with the provided data during the number of epochs

        Args:
            epochs (int): number of train cycles through the train data

        Return:
            loss (list): list of loss in each epoch, len(loss) = epochs

        """

        loss = list()
        for epoch in range(epochs):
            self.sess.run(self.use_train_data)
            for mini_batch in range(math.ceil(self.batch_size / self.mini_batch_size)): 
                self.sess.run([self.loss, self.train_op])

            if(epoch % 5 == 0):
                s = self.sess.run(self.summary)
                self.sum_writer.add_summary(s, self.epochs)

            self.epochs += 1
            self.sess.run(self.use_test_data)
            epoch_loss = self.sess.run(self.loss)
            loss.append(epoch_loss)
            
        return loss


    def predict(self, input=None):
        """
        Provides the input to the model and make inference for that input, returning the predicted value

        Args:
            input (list): list of values that will be provided to the model, must have shape = [batch_size, input_size]

        Return:
            output (list): list of values predicted by the model, will have shape = [batch_size, output_size]

        Raises:
            ValueError: input must be feed
            TypeError: input shape must be = [batch_size, input_size]

        """
        if(input!=None):
            if(len(np.shape(input)) == 2 and len(input[0]) == self.input_size):
                output = self.sess.run(self.output, {self.input:input})
                return output
            else:
                raise TypeError("input shape must be = [batch_size, input_size]")
        else:
            raise ValueError("input must be feed")
        


