import torch

from rn3ext import RWKV_TimeMix, Lru

n = 128
b = 2
d = 64
x1 = torch.randn(b, n, d).cuda()
x2 = torch.randn(b, n, d).cuda()
rwkv = RWKV_TimeMix(n, d, d, 3, 0).cuda()
lru = Lru(d).cuda()

rwkv_out = rwkv(x1)
lru_out = lru(x2)
lru_out_naive = lru.forward_naive(x2)

print(x1.shape, rwkv_out.shape)
print(x2.shape, lru_out.shape)
print(torch.norm(lru_out - lru_out_naive))
