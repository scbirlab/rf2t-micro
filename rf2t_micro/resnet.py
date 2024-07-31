"""Code 2D-convolutional residual blocks and networks."""

from typing import Iterable, Optional

import torch
from torch.nn import Conv2d, ELU, Dropout, InstanceNorm2d, Module, Sequential

def _get_same_padding(kernel: int, dilation: int):
    return (kernel + (kernel - 1) * (dilation - 1) - 1) // 2


class ResBlock2D(Module):
    """Original resblock.
    
    """
    def __init__(self, n_c, 
                 kernel: int = 3, 
                 dilation: int = 1, 
                 p_drop: float = 0.15):
        super().__init__()
        padding = _get_same_padding(kernel, dilation)

        layer_s = [
            Conv2d(n_c, n_c, kernel, padding=padding, dilation=dilation, bias=False),
            InstanceNorm2d(n_c, affine=True, eps=1e-6),
            ELU(inplace=True),
            Dropout(p_drop),
            Conv2d(n_c, n_c, kernel, dilation=dilation, padding=padding, bias=False),
            InstanceNorm2d(n_c, affine=True, eps=1e-6)
        ]
        self.layer = Sequential(*layer_s)
        self.final_activation = ELU(inplace=True)

    def forward(self, x):
        return self.final_activation(x + self.layer(x))


class ResBlock2D_bottleneck(Module):
    """Pre-activation bottleneck resblock.

    """
    def __init__(self, n_c, 
                 kernel: int = 3, 
                 dilation: int = 1, 
                 p_drop: float = 0.15):
        super().__init__()
        padding = _get_same_padding(kernel, dilation)
        n_b = n_c // 2 # bottleneck channel
        
        layer_s = [
            InstanceNorm2d(n_c, affine=True, eps=1e-6),
            ELU(inplace=True),
            Conv2d(n_c, n_b, 1, bias=False),
            InstanceNorm2d(n_b, affine=True, eps=1e-6),
            ELU(inplace=True),
            Conv2d(n_b, n_b, kernel, dilation=dilation, padding=padding, bias=False),
            InstanceNorm2d(n_b, affine=True, eps=1e-6),
            ELU(inplace=True),
            Dropout(p_drop),
            Conv2d(n_b, n_c, 1, bias=False)
        ]
        self.layer = Sequential(*layer_s)

    def forward(self, x):
        return x + self.layer(x)


class ResidualNetwork(Module):
    def __init__(self, n_block, n_feat_in, n_feat_block, n_feat_out, 
                 dilation: Optional[Iterable[int]] = None, 
                 block_type: str = 'orig', 
                 p_drop: float = 0.15):
        super().__init__()
        if dilation is None:
            dilation = [1,2,4,8]

        # project to n_feat_block
        if n_feat_in != n_feat_block:
            layer_s = [Conv2d(n_feat_in, n_feat_block, 1, bias=False)]
            if block_type == 'orig': # should activate input
                layer_s += [InstanceNorm2d(n_feat_block, affine=True, eps=1e-6),
                            ELU(inplace=True)]
        else:
            layer_s = []

        # add resblocks
        for i_block in range(n_block):
            d = dilation[i_block % len(dilation)]
            if block_type == 'orig':
                res_block = ResBlock2D(n_feat_block, kernel=3, dilation=d, p_drop=p_drop)
            else:
                res_block = ResBlock2D_bottleneck(n_feat_block, kernel=3, dilation=d, p_drop=p_drop)
            layer_s.append(res_block)

        if n_feat_out != n_feat_block:
            # project to n_feat_out
            layer_s.append(Conv2d(n_feat_block, n_feat_out, 1))
        
        self.layer = Sequential(*layer_s)
    
    def forward(self, x):
        return self.layer(x)

