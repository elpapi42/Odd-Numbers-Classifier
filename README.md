# Odd-Numbers-Classifier
Neural Network Classifier Model setup as an Odd/Even Numbers Classifier

In this student project, for practice what i self learned about Tensorflow in 1 month, i use the Classifier model, for classify numbers into Odd or Even Numbers.
You can interact with this model application using a simple console menu system, and play with the model parameters for see how it react to different setups.
The user can edit stuff like learning rate, mini-batch size, training samples, model topology and more.

The Odd number classifier was built using a custom adaptable model created using tensorflow from scratch 
(No high level tensorflow features was used, only tf.Optimizers). this model can be used for classify data (like statistics) into several classes
The model use tf.summary for register data and visualize it on Tensorboard

The core of the model is built in three python classes, contained in Classes/ folder. 
I plan put this simple model in an standalone repository for any one that want to use it for his own project and his own purposes

The model was created from low level Tensorflow API (Tensorflow Core) and Tensorflow Optimizers, the next python modules was used for create this:

- tensorflow-gpu 1.13.1
- bitstring 3.1.5
- console-menu 0.5.1 (Simple UI for interact with the model parameters)
- numpy 1.16.3

This is my first aproximation to the tensorflow tools, and this can still have a wide margin for improvement. 
the project is open to collaborations, if you want to callaborate, just fork and send your PR, any improvement is welcome and will help to learn more