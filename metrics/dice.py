import sys
import torch
import torch.nn.functional as F

def Dice(inp, gt):

    F.softmax(inp)
    return

def OnlineDice(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183

    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = torch.sum(iflat * tflat)

    A_sum = torch.sum(iflat)
    B_sum = torch.sum(tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

def OnlineDice3(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183

    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    print(f"dice3: {pred.shape}")
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    dims = pred.shape[2:]
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = torch.sum(iflat * tflat, dims)

    A_sum = torch.sum(iflat, dims)
    B_sum = torch.sum(tflat, dims)

    return 1 - torch.sum((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def OnlineDice2(input,target):
    """
    https://github.com/rogertrullo/pytorch/blob/rogertrullo-dice_loss/torch/nn/functional.py#L708

    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques=torch.unique(target)
    assert set(uniques.detach().tolist())<=set([0,1]), "target must only contain zeros and ones"

    probs=F.softmax(input, 1)
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)


    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)


    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c


    dice=2*(num/(den1+den2))
    dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

    dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

    return dice_total

def DiceWin(inp, gt, smooth=1., e=1.0e-8):
    """
    Inspired by Winson's dice
    """
    dims = list(range(2, len(inp.shape)))
    return -2 * torch.mean((torch.sum(inp * gt, dims) + smooth) / (torch.sum(inp, dims) + torch.sum(gt, dims) + smooth + e))

def DiceMax(inp, gt, smooth=1., e=1.0e-8):
    """
    Max's dice
    """
    dims = list(range(2, len(inp.shape)))
    valid=~torch.eq(gt,105)
    return - 2 * torch.mean((torch.sum(inp*gt*valid, dims)+smooth)/(torch.sum(inp*valid, dims)+torch.sum(gt*valid, dims)+smooth+e))
