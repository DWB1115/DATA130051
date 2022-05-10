import numpy as np
import torch 


def mixup_loss(criterion, pred, y_a, y_b, lamda):
    return lamda * criterion(pred, y_a) + (1 - lamda) * criterion(pred, y_b)

def mixup_data(X, y, alpha=0.2, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lamda = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = X.size()[0]
    index = torch.randperm(batch_size).cuda() if use_cuda else torch.randperm(batch_size)

    mixed_x = lamda * X + (1 - lamda) * X[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lamda
