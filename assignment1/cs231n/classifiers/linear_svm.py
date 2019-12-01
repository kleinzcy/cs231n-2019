from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    # initialize the gradient as zero
    dW = np.zeros(W.shape)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            # note delta = 1
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            # note delta = 1
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # average by num_train
    dW /= num_train

    # add the regularization to dW
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0

    # initialize the gradient as zero
    dW = np.zeros(W.shape)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Compute the score and extract the correct class according to y
    num_train = X.shape[0]
    num_class = W.shape[1]
    scores = X.dot(W)
    correct_class_scores = scores[range(num_train), y].reshape(-1, 1)

    # Compute the margin and filter the margin small than 0
    margin = scores - correct_class_scores + 1
    margin_beyond_zero = np.where(margin > 0, 1, 0)
    margin *= margin_beyond_zero

    # Compute the mean loss and add the regularization.
    # Notice we subtract 1 behind the loss, because we add 1 loss in the correct class.
    loss += margin.sum()/num_train + reg * np.sum(W * W) - 1
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    y_one_hot = np.zeros((num_train, num_class))
    y_one_hot[range(num_train), y] = 1
    dW += X.T.dot(margin_beyond_zero) - (margin_beyond_zero.sum(axis=1).reshape(-1, 1)*X).T.dot(y_one_hot)

    dW /= num_train
    # add regu
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
