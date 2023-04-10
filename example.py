import torch

from rnn import RWKV_TimeMix, Lru

n = 10
b = 2
d = 64
x = torch.randn(b, n, d).cuda()
rwkv = RWKV_TimeMix(n, d, d, 3, 0).cuda()
lru = Lru(d).cuda()

rwkv_out = rwkv(x)
lru_out = lru(x.transpose(1, 0))
lru_out_naive = lru.forward_naive(x.transpose(1, 0))

print(x.shape, rwkv_out.shape)
print(lru_out.shape)
print(torch.norm(lru_out - lru_out_naive))