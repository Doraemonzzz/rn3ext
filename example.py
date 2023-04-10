import torch

from rnn import RWKV_TimeMix

n = 10
b = 2
d = 32
x = torch.randn(b, n, d).cuda()
rwkv = RWKV_TimeMix(n, d, d, 3, 0).cuda()

rwkv_out = rwkv(x)

print(x.shape, rwkv_out.shape)
