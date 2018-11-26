import torch
a = torch.randn(5, 3, 8, 8)

q, w, e = torch.split(a, 2, dim=0)
# q : 2 3 8 8
# w : 2 3 8 8
# e : 1 3 8 8 

# G : 2, M : 3
              # G  M  C  H  W
b = torch.zeros(2, 3, 3, 8, 8)
b = 