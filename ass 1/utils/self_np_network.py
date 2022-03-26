from builtins import range
from builtins import object
import numpy as np
import pickle



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None

    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_ravel = np.ravel(x).reshape(N, D) 
    out = x_ravel.dot(w) + b

    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_ravel = np.ravel(x).reshape(N, D) 
    dx = dout.dot(w.T).reshape(x.shape)
    dw = (x_ravel.T).dot(dout)
    db = np.sum(dout, axis=0)
    
    return dx, dw, db

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
   
    out = np.maximum(0, x)
    
    cache = x
    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = (x > 0) * dout

    return dx

def affine_relu_forward(x, w, b):
    """
      Convenience layer that perorms an affine transform followed by a ReLU

      Inputs:
      - x: Input to the affine layer
      - w, b: Weights for the affine layer

      Returns a tuple of:
      - out: Output from the ReLU
      - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def svm_loss(x, y):
    """
      Computes the loss and gradient using for multiclass SVM classification.

      Inputs:
      - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
        class for the ith input.
      - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
        0 <= y[i] < C

      Returns a tuple of:
      - loss: Scalar giving the loss
      - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    
    N = x.shape[0]
    x_index = np.array(range(N))    # shape : (N,)
    correct_score_arr = x[x_index, y].reshape(N, 1)   # shape : (N,1)
    margin = np.maximum(0, x - correct_score_arr + 1)
    margin[x_index, y] = 0.0
    loss = float(np.sum(margin)) / N
    dx = (margin > 0) * 1.0
    dx[x_index, y] -= 1.0 * np.sum(margin > 0, axis=1)
    dx /= N
    
    return loss, dx

def softmax_loss(x, y):
    """
      Computes the loss and gradient for softmax classification.

      Inputs:
      - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
        class for the ith input.
      - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
        0 <= y[i] < C

      Returns a tuple of:
      - loss: Scalar giving the loss
      - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    
    x = x.astype(np.float64) 
    N = x.shape[0]
    x_shift = x - np.max(x, axis=1).reshape(-1, 1)
    x_index = np.array(range(N))    # shape : (N,)
    softmax_output = np.exp( x_shift ) / np.sum(np.exp( x_shift ), axis=1).reshape(-1, 1)
    loss=  - 1.0 * np.sum( np.log(1e-100 + softmax_output[x_index, y]))
    loss /=  N
    dx = softmax_output
    dx[x_index, y] -= 1.0
    dx /= N
    
    return loss, dx


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=28 * 28,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        N = X.shape[0]
        out1, (cache1, cache2) = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache3 = affine_forward(out1, self.params['W2'], self.params['b2'])
        if y is None:
            return scores

        loss, grads = 0, {}
      
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * ( np.sum(self.params['W2'] * self.params['W2']) \
                                 + np.sum(self.params['W1'] * self.params['W1']))
        dout2, grads['W2'], grads['b2'] = affine_backward(dout, cache3)
        grads['W2'] += self.reg * self.params['W2']

        _, grads['W1'], grads['b1'] = affine_relu_backward(dout2, (cache1, cache2))
        grads['W1'] += self.reg * self.params['W1']

        return loss, grads

    def save(self,path):
        obj = pickle.dumps(self)
        with open(path,"wb") as f:
            f.write(obj)

    def load(path):
        obj = None
        with open(path, "rb") as f:
            try:
                obj = pickle.load(f)
            except:
                print("IOError")
        return obj