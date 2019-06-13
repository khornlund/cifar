import torch


def accuracy(output, target):
    with torch.no_grad():
      correct = (output == target).type(torch.FloatTensor)
      return correct.mean()
