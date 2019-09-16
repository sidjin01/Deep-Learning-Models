#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 02:22:03 2019

@author: siddjin
"""
import numpy as np

# auxiliary functions
def relu(x):
    return np.maximum(x, 0)


def der_relu(x):
    der = np.zeros_like(x)
    der[x > 0] = 1
    # x[x <= 0] = 0
    return der


def softmax(x):
    if x.max() > 0:
        x -= x.max()
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=1)


def create_mask_from_window(x):
    """
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    mask = x == np.max(x)
    return mask


def conv_forward(A_prev, W, b, stride):
    """
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev)
    W -- Weights, numpy array of shape (f, f, n_C)
    b -- Biases, numpy array of shape (1, 1, n_C)
    Returns:
    A_next -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    (m, n_H_prev, n_W_prev) = A_prev.shape
    (f, f, n_C) = W.shape
    n_H = int((n_H_prev - f) // stride + 1)
    n_W = int((n_W_prev - f) // stride + 1)
    Z = np.zeros((m, n_H, n_W, n_C))
    for h in range(n_H):  # loop over vertical axis of the output volume
        for w in range(n_W):  # loop over horizontal axis of the output volume
            vert_start = h * stride
            vert_end = h * stride + f
            horiz_start = w * stride
            horiz_end = w * stride + f
            A_slice_prev = A_prev[
                :, vert_start:vert_end, horiz_start:horiz_end, np.newaxis
            ]
            Z[:, h, w, :] = (
                np.sum(A_slice_prev * W[np.newaxis, :, :, :], axis=(1, 2)) + b
            )
    assert Z.shape == (m, n_H, n_W, n_C)

    A_next = relu(Z)
    # parameters_conv= (W,b)
    cache_conv = (A_next, W, b)
    return cache_conv


def pool_forward(A_prev, pool_size):
    """
    Implements the forward pass of the pooling layer
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_next, n_W_next, n_C)
    Returns:
    A_next -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters
    """
    (m, n_H_prev, n_W_prev, n_C) = A_prev.shape
    n_H = int(1 + (n_H_prev - pool_size))
    n_W = int(1 + (n_W_prev - pool_size))
    A_next = np.zeros((m, n_H, n_W, n_C))
    for h in range(n_H):  # loop on the vertical axis of the output volume
        for w in range(
            n_W
        ):  # loop on the horizontal axis of the output volume
            vert_start = h
            vert_end = h + pool_size
            horiz_start = w
            horiz_end = w + pool_size
            A_next_slice = A_prev[
                :, vert_start:vert_end, horiz_start:horiz_end, :
            ]
            A_next[:, h, w, :] = np.max(A_next_slice, axis=(1, 2))
    assert A_next.shape == (m, n_H, n_W, n_C)
    cache_pool = (A_prev, A_next)
    return cache_pool


def multilayer(A_prev, parameters):
    W1 = parameters["W1"]  # dimemnsions(n_H*n_W*n_C,h)
    b1 = parameters["b1"]  # dimensions(m,h)
    W2 = parameters["W2"]  # dimensions(h,10)
    b2 = parameters["b2"]  # dimensions(m,10)
    (m, n_H, n_W, n_C) = A_prev.shape
    A_prev = A_prev.reshape(m, n_H * n_W * n_C)
    Z = np.dot(A_prev, W1) + b1
    Z = relu(Z)
    A_next = np.dot(Z, W2) + b2
    A_next = softmax(A_next)
    cache_nn = (A_prev, Z, A_next)
    return cache_nn


def cost(cache_nn, y_train):
    now_cost = 0
    X, Z, Y = cache_nn
    m, n = y_train.shape
    for row in range(m):
        now_cost -= np.dot(y_train[row], np.transpose(np.log(Y[row])))
    return now_cost / m


def backprop(parameters, cache_nn, y_train):
    (X, Z, Y) = cache_nn
    W2 = parameters["W2"]
    W1 = parameters["W1"]
    Y_grad = Y - y_train
    W2_grad = np.dot(Z.T, Y_grad)
    b2_grad = np.mean(Y_grad, axis=0, keepdims=1)
    Z_grad = np.dot(Y_grad, W2.T)
    Z_grad = Z_grad * der_relu(Z)
    W1_grad = np.dot(X.T, Z_grad)
    b1_grad = np.mean(Z_grad, axis=0, keepdims=1)
    dA_prev = np.dot(Z_grad, W1.T)
    gradients = {
        "W2_grad": W2_grad,
        "b2_grad": b2_grad,
        "W1_grad": W1_grad,
        "b1_grad": b1_grad,
    }
    return dA_prev, gradients


def pool_backward(dA, A_prev):
    """
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    for h in range(n_H):  # loop on the vertical axis
        for w in range(n_W):  # loop on the horizontal axis
            vert_start = h
            vert_end = h + n_H
            horiz_start = w
            horiz_end = w + n_W
            A_prev_slice = A_prev[
                :, vert_start:vert_end, horiz_start:horiz_end, :
            ]
            mask = create_mask_from_window(A_prev_slice)
            K = np.multiply(mask, dA[:, h, w, :][:, np.newaxis, np.newaxis, :])
            dA_prev[:, vert_start:vert_end, horiz_start:horiz_end, :] += K
    assert dA_prev.shape == A_prev.shape
    return dA_prev


def conv_backward(dZ, A_prev, stride):
    """
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, n_C)
    """
    (m, n_H_prev, n_W_prev) = A_prev.shape
    (f, f, n_C) = W.shape
    (m, n_H, n_W, n_C) = dZ.shape
    dA_prev = np.zeros((m, n_H_prev, n_W_prev))
    dW = np.zeros((f, f, n_C))
    db = np.zeros((1, 1, n_C))
    for h in range(n_H):  # loop over vertical axis of the output volume
        for w in range(n_W):  # loop over horizontal axis of the output volume
            vert_start = h * stride
            vert_end = h * stride + f
            horiz_start = w * stride
            horiz_end = w * stride + f
            A_slice = A_prev[
                :, vert_start:vert_end, horiz_start:horiz_end, np.newaxis
            ]
            # da_prev[vert_start:vert_end, horiz_start:horiz_end] += W[:,:,c] * dZ[i, h, w, c]
            dW += np.mean(
                A_slice * dZ[:, h, w, :][:, np.newaxis, np.newaxis, :], axis=0
            )
            db += np.mean(dZ[:, h, w, :], axis=0)
    assert dA_prev.shape == (m, n_H_prev, n_W_prev)
    return dW, db


def CNN(
    X, W, b, y_train, parameters, pool_size, stride, alpha, epochs, batch_size
):
    for i in range(epochs):
        cost_sum = 0
        for j in range(0, len(X), batch_size):
            batch_x = X[j : j + batch_size]
            batch_y = y_train[j : j + batch_size]
            # goimng forward
            cache_conv = conv_forward(batch_x, W, b, stride)
            A1, W, b = cache_conv
            cache_pool = pool_forward(A1, pool_size)
            A1, A2 = cache_pool
            m, h, w, c = A2.shape
            cache_nn = multilayer(A2, parameters)
            A2, A3, A4 = cache_nn
            # computing cost
            cost_sum += cost(cache_nn, batch_y)
            if (j + batch_size) % 10 == 0:
                print(
                    "\repoch {:3d}: batch {:2d}: cost: {:.4f}".format(
                        i, j // batch_size, cost_sum / (j // batch_size + 1)
                    ),
                    end="",
                )
            # going backwards
            dA2, gradients = backprop(parameters, cache_nn, batch_y)
            dA2 = dA2.reshape(m, h, w, c)
            dA1 = pool_backward(dA2, A1)
            dW, db = conv_backward(dA1, batch_x, stride)
            # updating the gradients
            parameters["W1"] -= alpha * gradients["W1_grad"]
            parameters["W2"] -= alpha * gradients["W2_grad"]
            parameters["b1"] -= alpha * gradients["b1_grad"]
            parameters["b2"] -= alpha * gradients["b2_grad"]
            W -= alpha * dW
            b -= alpha * db
            # decay alpha
            steps = i * (len(X) // batch_size) + (j // batch_size)
            if steps % 1000 == 0:
                alpha /= 2.0
    print("")
    return W, b, parameters


def test_result(X_test, W, b, parameters, y_test,batch_size):
    acc_sum = 0
    y_pred = 0
    for j in range(0, len(X_test), batch_size):
        batch_x = X_test[j : j + batch_size]
        batch_y = y_test[j : j + batch_size]
        # goimng forward
        cache_conv = conv_forward(batch_x, W, b, stride)
        A1, W, b = cache_conv
        cache_pool = pool_forward(A1, pool_size)
        A1, A2 = cache_pool
        m, h, w, c = A2.shape
        cache_nn = multilayer(A2, parameters)
        A2, A3, A4 = cache_nn
        y_pred+=A4
        acc_sum += 1 - np.sum(abs(A4 - batch_y) / batch_y.size)
        if j % 10 == 0:
            print("\rbatch {:3d}: ".format(j // batch_size + 1), end="")
            print("Accuracy is: {:.2f}", acc_sum / (j // batch_size + 1), end="")
    print("")
    return y_pred


np.random.seed(42)
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

from keras.utils import to_categorical

# one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
X_train = np.float32(X_train)
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = np.float32(X_test)
X_test = (X_test - X_test.mean()) / X_test.std()

W_shape = (3, 3, 16)
W_limit = np.sqrt(6 / (np.prod(W_shape[:2]) + W_shape[-1]))
W = np.random.uniform(low=-W_limit, high=W_limit, size=W_shape)
W1_shape = (2304, 256)
W1_limit = np.sqrt(6 / sum(W1_shape))
W1 = np.random.uniform(low=-W1_limit, high=W1_limit, size=W1_shape)
W2_shape = (256, 10)
W2_limit = np.sqrt(6 / sum(W2_shape))
W2 = np.random.uniform(low=-W2_limit, high=W2_limit, size=W2_shape)
b = np.zeros((1, 1, 16))
b1 = np.zeros((1, 256))
b2 = np.zeros((1, 10))
parameters = {}
parameters["W1"] = W1
parameters["W2"] = W2
parameters["b1"] = b1
parameters["b2"] = b2
pool_size = 2
stride = 2
epochs = 100
alpha = 2e-4
batch_size = 16
W_new, b_new, parameters_new = CNN(
    X_train[:30000],
    W,
    b,
    y_train[:30000],
    parameters,
    pool_size,
    stride,
    alpha,
    epochs,
    batch_size,
)
y_pred=test_result(X_test, W_new, b_new, parameters_new, y_test,batch_size)
