# Implementing and Comparing Activation functions
## Goals and Objectives :
Neural network activation functions are a crucial component of deep learning. 
Activation functions determine the output of a deep learning model, its accuracy, and also the computational efficiency of training a model—which can make or break a large scale neural network. 
Activation functions also have a major effect on the neural network’s ability to converge and the convergence speed.

## We are working with 6 activation funvtions: 
	1. Sigmoid
	2. tanH
	3. ReLU
	4. Leaky ReLU
	5. Swish
	6. Swish Beta

## Model: Convolutional Neural Network
To implement and compare all of the above activation functions we are using same CNN models to train and evaluate the image data.
The layers in model are:
	1. Input Layer (input_layer)
	2. Convolutional Layer (conv1)
	3. Pooling Layer (pool1)
	4. Normalization Layer (norm1)
	5. Convolutional Layer (conv2)
	6. Pooling Layer (pool2)
	7. Flatten Layer (pool2_flat)
	8. Fully Connected/ Dense Layer (dense)
	9. Logits/Output Layer (logits)

## Tensorflow Estimator
An Estimator is a TensorFlow class for performing high-level model training, evaluation, and inference for our model. 
The models saved in local directory and can be used later for training, evaluation, and inference even after kernel reset or reconnecting your server.

## How to Run
When working on large dataset its convinient to use cloud services such as AWS, GCP, Google Colab etc.
Out of those three the google coolab is free to use and i have executed my code on google colab.
	### Guidlines to run this project on colab:
		1. Open google colab at https://colab.research.google.com/notebooks/welcome.ipynb and login
		2. Create a new notebooke or you can upload your jupyter notebook from file menu
		3. Go to Runtime Menu > Change Runtime type > A popup window will open. Select select runtime type 'python 3' and hardware accelerator 'GPU'
		4. Download dataset from https://www.kaggle.com/puneet6060/intel-image-classification
		5. Keep only 'buildings' and 'forest' data in Train and Test set and remove other
		6. Compressed Train and Test image Dataset in .zip formate 
		7. Upload the the zipped folder to colab using Upload button provided in Files section
		8. Uncomment the "Unzip Uploaded file on colab" cell code and run entire notebook
	To learn Google colab basics, please refer :https://www.youtube.com/watch?time_continue=1&v=inN8seMm7UI
	
## Implementing analysis.
Getting Accuracy, loss, execution time for each activation function and comparing them.