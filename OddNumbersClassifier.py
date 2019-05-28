#Calculates the propabilities of the given number for be an Odd Number
#Whitman Bohorquez / ElPapi42
#Sample project for practice what i self-learned about tensorflow

import tensorflow as tf
import numpy as np
from Classes.Classifier import Classifier as cl
from bitstring import BitArray
from consolemenu import *
from consolemenu.items import *




class ClassifierMenu(object):
    """Contains the User Interface Menus"""

    def __init__(self):

        #Parameters
        self.bit_lenght = 32
        self.learning_rate = 0.001
        self.mini_batch_size = 100
        self.model_shape = [1]
        self.train_samples = 500
        self.epochs = 100

        #Main Menu
        self.main_menu = ConsoleMenu("Odd Numbers Classifier", "Classify numbers in two classes, Odd or Even")

        #Parameters Menu
        self.params_menu = ConsoleMenu("Set Parameters", "Adjust the parameters of the model",
                                       prologue_text="When one of this parameters is changed the model is regenerated and the training done before is loss")
        self.params_menu.append_item(FunctionItem("Set Learning Rate -> Current: " + str(self.learning_rate), self.lr_set))
        self.params_menu.append_item(FunctionItem("Set Mini Batch Size -> Current: " + str(self.mini_batch_size), self.mbs_set))
        self.params_menu.append_item(FunctionItem("Set Model Shape -> Current: " + str(self.model_shape), self.ms_set))
        self.params_menu.append_item(FunctionItem("Set Number of Training Samples -> Current: " + str(self.train_samples), self.ts_set))

        #Train Menu
        self.train_menu = ConsoleMenu("Train The Model", "Model Performance(Go to tensorboard for detailed Data):", prologue_text="Model Error: Unknow")
        self.train_menu.append_item(FunctionItem("Set Epochs Per Train -> Current: " + str(self.epochs), self.epochs_set))
        self.train_menu.append_item(FunctionItem("Perform Train", self.perform_train))

        #Inference Menu
        self.inference_menu = ConsoleMenu("Inference", "Predict if the supplied number is Odd or Even", prologue_text="Result: Unknow")
        self.inference_menu.append_item(FunctionItem("Predict Value", self.predict))

        #Append Submenus to Main Menu
        self.main_menu.append_item(SubmenuItem("Set Parameters", self.params_menu))
        self.main_menu.append_item(SubmenuItem("Train Model", self.train_menu))
        self.main_menu.append_item(SubmenuItem("Inference", self.inference_menu))

        #Generates model from default parameters
        self.gen_train_data()
        self.gen_model()

        #Show Main Menu
        self.main_menu.show()

    def gen_model(self):

        tf.reset_default_graph()

        self.model = cl(self.train_data_in, self.train_data_out, self.model_shape, self.mini_batch_size, self.learning_rate)
        self.train_menu.prologue_text = "Model Error: Unknow"

    def gen_train_data(self):

        #Generates Train Data
        rand_list = np.random.randint(-pow(2, self.bit_lenght - 1), pow(2, self.bit_lenght - 1), self.train_samples)
        self.train_data_in = [list(np.array(list(np.binary_repr(n, self.bit_lenght))).astype(np.int)) for n in rand_list]
        self.train_data_out = [[float(BitArray(n).int % 2), float(not(bool(BitArray(n).int % 2)))] for n in self.train_data_in]

    def lr_set(self):
        
        print("Learning Rate: Represent the lenght of the step of the gradient during training phase\n" + 
                "Higher values mean faster learning, but increases the probabilities of never get into a minimum\n" +
                "Lower values mean accurate learning, but increases the probabilities of never get out of local minimuns\n")
        print("New Learning Rate: ")

        try:
            new_rate = float(input())
        except:
            print("Invalid Rate")
            input()

        if(new_rate > 0.0):
            self.learning_rate = new_rate
            self.params_menu.items[0].text = "Set Learning Rate -> Current: " + str(self.learning_rate)
            self.gen_model()
        else:
            print("Learning Rate Must Be > 0")
            input()
         
    def mbs_set(self):
        
        print("Mini Batch Size: Represent the amount of training data items that will be feed to the model per training step\n" +
                "Higher values minimizes training data noise print, the model generalize better, but can produce overfitting\n" +
                "Lower values prevent overfitting, but produce lower learning speeds\n")
        print("New Mini Batch Size: ")

        try:
            new_size = int(input())
        except:
            print("Invalid Value")
            input()

        if(new_size >= 1):
            self.mini_batch_size = new_size
            self.params_menu.items[1].text = "Set Mini Batch Size -> Current: " + str(self.mini_batch_size)
            self.gen_model()
        else:
            print("Mini Batch Size Must Be >= 1")
            input()

    def ms_set(self):
        
        print("Model Shape: list of integers separated by whitespaces, if there is non-numeric characters, the shape will be non-valid\n" + 
                "The total number of elements in the list represent the number of layers\n" +
                "Each number in the list, represent the number of neurons in that layer\n" +
                "High amount of neurons are capable of learn complex information, but slow down the training and inference\n" +
                "Low amount of neurons can prevent the model for reach a accurate minimum\n" +
                "For this Odd/Even numbers classifier there is no need of a lot of neurons, only one layer with 1 neuron is capable of learn this data\n")
        print("New Model Shape: ")

        try:
            new_shape = np.array(input().rsplit(" ", -1)).astype(np.int)
        except:
            print("Invalid Shape")
            input()

        if(len(new_shape) >= 1 and np.min(new_shape) > 0):
            self.model_shape = new_shape
            self.params_menu.items[2].text = "Set Model Shape -> Current: " + str(self.model_shape)
            self.gen_model()
        else:
            print("All the numbers must be > 0")
            input()
        
    def ts_set(self):
        
        print("Number Of Training Samples: How many training samples will be generated\n" + 
                "Higer Amounts help the model generalize better, but slow down the training speed\n" +
                "Lower Amounts can guide the model to generalize bad, but improves training speed\n")
        print("New Number Of Training Samples: ")

        try:
            new_samples = int(input())
        except:
            print("Invalid Value")
            input()

        if(new_samples > 0):
            self.train_samples = new_samples
            self.params_menu.items[3].text = "Set Number of Training Samples -> Current: " + str(self.train_samples)
            self.gen_train_data()
            self.gen_model()
        else:
            print("Number of sample must be > 0")
            input()
        
    def epochs_set(self):
        
        print("Number Of Epochs: A single epoch represent a full cycle trought the traning data, more epochs mean more cycles\n" + 
                "Higer Amounts let the model train more time\n")
        print("New Number Of Epochs: ")

        try:
            new_epochs = int(input())
        except:
            print("Invalid Value")
            input()

        if(new_epochs > 0):
            self.epochs = new_epochs
            self.train_menu.items[0].text = "Set Epochs Per Train -> Current: " + str(self.epochs)
        else:
            print("Number of epochs must be > 0")
            input()
        
    def perform_train(self):
        results = self.model.train(self.epochs)
        self.train_menu.prologue_text = "Model Error: " + str(np.round(results[-1], 3))

    def predict(self):
        
        print("Provide a 32bit number: ")

        try:
            number = int(input())
        except:
            print("Invalid Value")
            input()

        bin_number = list(np.array(list(np.binary_repr(number, self.bit_lenght))).astype(np.int))
        prediction = self.model.predict([bin_number])
        most_probable = np.max(prediction[0])
        if(most_probable == prediction[0][0]):
            self.inference_menu.prologue_text = ("Results: " + str(number) + " Has " + str(np.round(most_probable * 100, 4)) + 
                                                "% Chances of be an Odd Number")
        else:
            self.inference_menu.prologue_text = ("Results: " + str(number) + " Has " + str(np.round(most_probable * 100, 4)) + 
                                                "% Chances of be an Even Number")
        

def main():
    cl_menu = ClassifierMenu()

if __name__ == "__main__":
    main()