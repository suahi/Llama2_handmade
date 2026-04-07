import torch


# define a RMSNorm function
# after a hidden_state, shape@[bs,seq,dim]
def RMSNorm(x, gamma, epsilon):
    bs, seq, dim = x.shape
    # todo:不熟悉torch对元素平方累加
    temp = torch.sum(x ** 2, dim=1)
    return x * gamma / torch.sqrt(temp / seq + epsilon)


def Attention():
    pass


def Decoder():
    pass


def MLP():
    pass


def Transformer():
    pass
