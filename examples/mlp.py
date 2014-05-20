#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sklearn.datasets
import nnet


def run():
    # Fetch data
    digits = sklearn.datasets.load_digits()
    X_train = digits.data
    X_train /= np.max(X_train)
    y_train = digits.target
    n_classes = np.unique(y_train).size

    # Setup multi-layer perceptron 
    nn = nnet.NeuralNetwork(
        layers=[
            nnet.Linear(
                n_out=50,
                weight_scale=0.1,
                weight_decay=0.002,
            ),
            nnet.Activation('relu'),
            nnet.Linear(
                n_out=n_classes,
                weight_scale=0.1,
                weight_decay=0.002,
            ),
            nnet.LogRegression(),
        ],
    )

    # Verify network for correct back-propagation of parameter gradients
    print('Checking gradients')
    nn.check_gradients(X_train[:100], y_train[:100])

    # Train neural network
    print('Training neural network')
    nn.fit(X_train, y_train, learning_rate=0.1, max_iter=25, batch_size=32)

    # Evaluate on training data
    error = nn.error(X_train, y_train)
    print('Training error rate: %.4f' % error)


if __name__ == '__main__':
    run()
