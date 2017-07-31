import math

import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  dims = X.shape[1]
  num_classes = W.shape[1]
  logits = X.dot(W)
  logits -= np.max(logits, axis=1, keepdims=True)
  for example in xrange(num_train):
    norm = sum(math.exp(logit) for logit in logits[example])
    probabilities = [math.exp(logit) / norm for logit in logits[example]]
    loss -= math.log(probabilities[y[example]]) / num_train
    for dim in xrange(dims):
      for cls in xrange(num_classes):
        dW[dim, cls] += 1.0 / num_train * X[example, dim] * (
            probabilities[cls] - (1 if cls == y[example] else 0))
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  logits = X.dot(W)
  logits -= np.max(logits, axis=1, keepdims=True)
  scores = np.exp(logits)
  probabilities = scores / np.sum(scores, axis=1, keepdims=True)
  loss = -np.sum(np.log(probabilities[np.arange(num_train), y])) / num_train
  loss += reg * np.sum(W * W)
  labels_onehot = np.zeros(probabilities.shape)
  labels_onehot[np.arange(num_train), y] = 1
  dW = X.T.dot(probabilities - labels_onehot) / num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

