import torch
import math
from model_nerv import Generator

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

model = Generator(embed_length=80, stem_dim_num='512_1', fc_hw_dim='9_16_26', expansion=1.0, 
        num_blocks=1, norm='none', act='swish', bias = True, reduction=2, conv_type='conv',
        stride_list=[5, 2, 2, 2, 2],  sin_res=True,  lower_width=96, sigmoid=False)

# for k, v in model.named_parameters():
#     print(k)

# for p in model.parameters():
#     print(p.shape)

#torch.save( model.state_dict(), './temp.pth' )

model_dict = model.state_dict()
print( model_dict.keys() )

mlp_dict = {}
for k, v in model_dict.items():
    if 'stem' in k:
        print(k)
        mlp_dict[k] = v

print("----")
body_dict = {}
for k, v in model_dict.items():
    if 'layers' in k and 'head_layers' not in k:
        print(k)
        body_dict[k] = v

print("----")
tail_dict = {}
for k, v in model_dict.items():
    if 'head_layers' in k:
        print(k)
        tail_dict[k] = v

hidden_list = [80, 512, 1024, 2048]

layers = []
lastv = hidden_list[0]

for hidden in hidden_list[1:-1]:
    layers.append( (lastv, hidden) )
    layers.append( 'ReLU')
    lastv = hidden
layers.append( (lastv, hidden_list[-1]))
print( layers )