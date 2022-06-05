import os
import cv2
import numpy as np
from sys import path
from Network.NN import NN
from Network.Layer import Layer
from utils.data_loading import Dataset
from sys import path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
Data = Dataset().load('D:/Machine_Learning/Pima-Indians-Diabetes-Dataset/pima-indians-diabetes.csv')
data = Dataset().preprocess(Data, mode='mean')

feature, targets = data[:, :-1], data[:, -1].reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(feature, targets, test_size=0.2)
# create the network
nn_model = NN(X_train, Y_train)
nn_model.add_layer(Layer(10, activation='leaky_relu'))
nn_model.add_layer(Layer(10, activation='leaky_relu'))

#fit the networ k
nn_model.fit(iteration=10000, learning_rate=0.001)

# plot cost function

Y_train_pred = nn_model.predict(X_train)
Y_test_pred = nn_model.predict(X_test)

print(accuracy_score(Y_test_pred, Y_test))
