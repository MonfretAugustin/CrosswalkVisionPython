import torch.nn as nn

def my_loss(regression, points):
    #regression is an array of predicted coordinates
    #points is an array of ground truth coordinates
    MSE = nn.MSELoss()
    return MSE(regression, points)