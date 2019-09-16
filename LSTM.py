#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:42:59 2019

@author: sid
"""
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_grad(x):
    return x*(1-x)
def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1,keepdims=1)
def der_tanh(x):
    return (1-np.square(np.tanh(x)))

def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)                       
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    cct stands for the candidate value (c tilde),
    c stands for the memory value
    """
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]
    n_x, m = xt.shape
    n_y, n_a = Wy.shape
    concat = np.zeros((n_a+n_x,m))
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = xt
    
    forget = sigmoid(np.dot(Wf,concat)+bf)
    update = sigmoid(np.dot(Wi,concat)+bi)
    output = sigmoid(np.dot(Wo,concat)+bo)

    cct = np.tanh(np.dot(Wc,concat)+bc)
    
    c_next = forget*c_prev + update*cct
    a_next = output*np.tanh(c_next)
    yt_pred = softmax(np.dot(Wy,a_next)+by)
    
    cache = (a_next, c_next, a_prev, c_prev, forget, update, cct, output, xt, parameters)

    return a_next, c_next, yt_pred, cache

def lstm_cell_backward(yt, yt_pred, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).
    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass
    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """


    (a_next, c_next, a_prev, c_prev, forget, update, cct, output, xt, parameters) = cache
    
    Wf = parameters["Wf"]
    Wi = parameters["Wi"]
    Wo = parameters["Wo"]
    Wc = parameters["Wc"]
    n_x, m = xt.shape
    n_a, m = a_next.shape
    concat = np.zeros((n_a+n_x,m))
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = xt
    
    dyt_pred = yt_pred-yt
    da_next = np.dot(dyt_pred, parameters['Wy'])
    dWy = np.dot(dyt_pred, a_next)
    dby=np.sum(dyt_pred, axis=1, keepdims=1)
    doutput = da_next*np.tanh(c_next)
    doutput = doutput*sigmoid_grad(output)
    
    dc_next=da_next*output*(der_tanh(c_next))
    
    dcct = (da_next * output * der_tanh(c_next) + dc_next) * update * (1 - cct ** 2)
    dupdate = (da_next * output * der_tanh(c_next) + dc_next) * cct * (1 - update) * update
    dforget = (da_next * output * der_tanh(c_next) + dc_next) * c_prev * forget * (1 - forget)

    dc_prev = (da_next * output * der_tanh(c_next) + dc_next) * forget
  
    dWc = np.dot(dcct, concat.T)
    dbc = np.sum(dcct, axis=1, keepdims=1)
    dWo = np.dot(doutput, concat.T)
    dbo = np.sum(doutput, axis=1, keepdims=1)
    dWf = np.dot(dforget,concat.T)
    dbf = np.sum(dforget, axis=1, keepdims=1)
    dWi = np.dot(dupdate, concat.T)
    dbi = np.sum(dupdate, axis=1, keepdims=1)
    
    # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)
    da_prev = np.dot(Wf[:n_a, :], dforget) + np.dot(Wi[:n_a, :], update) + np.dot(Wo[:n_a, :], 
                    doutput) +np.dot(Wc[:n_a, :], dcct)
    dxt = None
    ### END CODE HERE ###
    
    # Save gradients in dictionary
    gradients = {'dWy': dWy, "dby": dby, "dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev,
                 "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

    return gradients