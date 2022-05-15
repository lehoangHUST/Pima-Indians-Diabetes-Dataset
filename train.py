import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sys import path
from Network.NN import NN
from Network.Layer import Layer
from utils.data_loading import Dataset
from sys import path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def run(args):
    file, config_nn, lr, = args.path, args.config_nn, args.learning_rate
    # Load dataset
    Data = Dataset().load(file)
    data = Dataset().preprocess(Data, mode='median')

    # Train test split
    feature, targets = data[:, :-1], data[:, -1].reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(feature, targets, test_size=0.2)

    # create the network
    nn_model = NN(X_train, Y_train)
    nn_model.add_layer(Layer(24, activation='relu'))
    nn_model.add_layer(Layer(12, activation='sigmoid'))

    # Train the network
    history = nn_model.fit(iteration=10000, learning_rate=0.01)

    # plot accuracy function
    plt.plot(history['epochs'], history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    Y_train_pred = nn_model.predict(X_train)
    Y_test_pred = nn_model.predict(X_test)

    print(accuracy_score(Y_test_pred, Y_test))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='D:/Machine_Learning/Pima-Indians-Diabetes-Dataset/pima-indians-diabetes.csv', type=str, help='path dataset')
    parser.add_argument('--config_nn', default=None, type=str,
                        help='Configure neural net')
    parser.add_argument('--learning_rate', type=float,
                        help='Learning rate of Neural Network')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)