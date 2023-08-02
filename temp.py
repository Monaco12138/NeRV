import torch
import math


# x = []
# for i in range(5):
#     x += [ [1] , [2] ]
# print(x)


# x = '1.25_5'
# x = x.lower()
# print(x)


# lbase, levels = [float(x) for x in x.split('_')]
# levels = int(levels)
# embed_length = 2 * levels
# print( levels, embed_length )

# def Encoding( pos ):
#     pe_list = []
#     for i in range(levels):
#         temp_value = pos * lbase **(i) * math.pi
#         pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
#     print(pe_list)
#     return torch.stack(pe_list, 1)

# x = torch.tensor( [0.5] )
# print(x.shape)
# embedding = Encoding( x )
# print( embedding.shape )
# print( embedding )

X = torch.arange(3*3).reshape(3,3)
print(X)
Y = torch.arange(3*3).reshape(3,3) + 9
print(Y)

print( torch.stack([X,Y], 0), torch.stack([X,Y], 0).shape )
print( torch.stack([X,Y], 1), torch.stack([X,Y], 1).shape )
print( torch.stack([X,Y], 2), torch.stack([X,Y], 2).shape )