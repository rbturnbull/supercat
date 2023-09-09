import torch
from torch import nn
from torch import Tensor
import numpy as np

@torch.jit.script
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """
    Center-crops the encoder_layer to the size of the decoder_layer,
    so that merging (concatenation) between levels/blocks is possible.
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.

    Taken from https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862
    """
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4:  # 2D
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                            ]
        elif encoder_layer.dim() == 5:  # 3D
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2),
                            ((ds[2] - es[2]) // 2):((ds[2] + es[2]) // 2),
                            ]
    return encoder_layer, decoder_layer


def Conv(*args, dim:int, **kwargs):
    if dim == 2:
        return nn.Conv2d(*args, **kwargs)
    if dim == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"dimension {dim} not supported")    
    

def BatchNorm(*args, dim:int, **kwargs):
    if dim == 2:
        return nn.BatchNorm2d(*args, **kwargs)
    if dim == 3:
        return nn.BatchNorm3d(*args, **kwargs)
    raise ValueError(f"dimension {dim} not supported")    
    

def ConvTranspose(*args, dim:int, **kwargs):
    if dim == 2:
        return nn.ConvTranspose2d(*args, **kwargs)
    if dim == 3:
        return nn.ConvTranspose3d(*args, **kwargs)
    raise ValueError(f"dimension {dim} not supported")    
    

def AdaptiveAvgPool(*args, dim:int, **kwargs):
    if dim == 2:
        return nn.AdaptiveAvgPool2d(*args, **kwargs)
    if dim == 3:
        return nn.AdaptiveAvgPool3d(*args, **kwargs)
    raise ValueError(f"dimension {dim} not supported")    
    

class ResBlock(nn.Module):
    """ 
    Based on
        https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448 
        https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """
    def __init__(self, dim:int, in_channels:int, out_channels:int, downsample:bool, kernel_size:int=3):
        super().__init__()
        
        # calculate padding so that the output is the same as a kernel size of 1 with zero padding
        # this is required to be calculated becaues padding="same" doesn't work with a stride
        padding = (kernel_size - 1)//2 
        
        if downsample:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, dim=dim)
            self.shortcut = nn.Sequential(
                Conv(in_channels, out_channels, kernel_size=1, stride=2, dim=dim), 
                BatchNorm(out_channels, dim=dim)
            )
        else:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dim=dim)
            self.shortcut = nn.Sequential()

        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dim=dim)
        self.bn1 = BatchNorm(out_channels, dim=dim)
        self.bn2 = BatchNorm(out_channels, dim=dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + shortcut
        return self.relu(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        dim:int,
        in_channels:int = 1,
        downsample:bool = True,
        growth_factor:float = 2.0,
        kernel_size:int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        if downsample:
            self.out_channels = int(growth_factor*self.out_channels)

        self.block1 = ResBlock(in_channels=in_channels, out_channels=self.out_channels, downsample=downsample, dim=dim, kernel_size=kernel_size)
        self.block2 = ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, downsample=False, dim=dim, kernel_size=kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        dim:int,
        in_channels:int,
        out_channels:int, 
        resblock_kernel_size:int = 3,
        upsample_kernel_size:int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upsample = ConvTranspose(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=upsample_kernel_size, stride=2, dim=dim)

        self.block1 = ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, downsample=False, dim=dim, kernel_size=resblock_kernel_size)
        # self.block2 = ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, downsample=False, dim=dim, kernel_size=resblock_kernel_size)

    def forward(self, x: Tensor, shortcut: Tensor) -> Tensor:
        x = self.upsample(x)
        # crop upsampled tensor in case the size is different from the shortcut connection
        x, shortcut = autocrop(x, shortcut)
        x += shortcut
        x = self.block1(x)
        # x = self.block2(x)
        return x


class ResNetBody(nn.Module):
    def __init__(
        self,
        dim:int,
        in_channels:int = 1,
        initial_features:int = 64,
        growth_factor:float = 2.0,
        kernel_size:int = 3,
        stub_kernel_size:int = 7,
        layers:int = 4,
    ):
        super().__init__()

        self.initial_features = initial_features
        self.in_channels = in_channels
        self.growth_factor = growth_factor
        self.kernel_size = kernel_size
        self.stub_kernel_size = stub_kernel_size
        self.layers = layers
        self.dim = dim

        current_num_features = initial_features
        padding = (stub_kernel_size - 1)//2
        
        self.stem = nn.Sequential(
            Conv(in_channels=in_channels, out_channels=current_num_features, kernel_size=stub_kernel_size, stride=2, padding=padding, dim=dim),
            BatchNorm(num_features=current_num_features, dim=dim),
            nn.ReLU(inplace=True),
        )

        self.downblock_layers = []
        for _ in range(layers):
            downblock = DownBlock( 
                in_channels=current_num_features, 
                downsample=True, 
                dim=dim, 
                growth_factor=growth_factor, 
                kernel_size=kernel_size 
            )
            self.downblock_layers.append(downblock)
            current_num_features = downblock.out_channels
        self.downblocks = nn.Sequential(*self.downblock_layers)
        self.output_features = current_num_features
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.downblocks(x)
        return x

    def macs(self):
        return resnetbody_macs(
            dim=self.dim,
            growth_factor=self.growth_factor,
            kernel_size=self.kernel_size,
            stub_kernel_size=self.stub_kernel_size,
            initial_features=self.initial_features,
            downblock_layers=self.layers,
        )


class ResNet(nn.Module):
    def __init__(
        self,
        dim:int,
        num_classes:int=1,
        body = None,
        in_channels:int = 1,
        initial_features:int = 64,
        growth_factor:float = 2.0,
        layers:int = 4,
    ):
        super().__init__()
        self.body = body if body is not None else ResNetBody(
            dim=dim, 
            in_channels=in_channels, 
            initial_features=initial_features, 
            growth_factor=growth_factor,
            layers=layers,
        )
        assert in_channels == self.body.in_channels
        assert initial_features == self.body.initial_features
        self.global_average_pool = AdaptiveAvgPool(1, dim=3)
        self.final_layer = torch.nn.Linear(self.body.output_features, num_classes)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.body(x)        
        # Final layer
        x = self.global_average_pool(x)
        x = torch.flatten(x, 1)
        output = self.final_layer(x)
        return output


class ResidualUNet(nn.Module):
    def __init__(
        self,
        dim:int,
        body:ResNetBody = None,
        in_channels:int = 1,
        initial_features:int = 64,
        out_channels: int = 1,
        growth_factor:float = 2.0,
        kernel_size:int = 3,
        downblock_layers:int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.body = body if body is not None else ResNetBody(
            dim=dim, 
            in_channels=in_channels, 
            initial_features=initial_features, 
            growth_factor=growth_factor,
            kernel_size=kernel_size,
            layers=downblock_layers,
        )
        assert in_channels == self.body.in_channels
        assert initial_features == self.body.initial_features

        self.upblock_layers = nn.ModuleList()
        for downblock in reversed(self.body.downblock_layers):
            upblock = UpBlock(
                dim=dim, 
                in_channels=downblock.out_channels, 
                out_channels=downblock.in_channels, 
                resblock_kernel_size=kernel_size
            )
            self.upblock_layers.append(upblock)

        self.final_upsample_dims = self.upblock_layers[-1].out_channels//2
        self.final_upsample = ConvTranspose(
            in_channels=self.upblock_layers[-1].out_channels, 
            out_channels=self.final_upsample_dims, 
            kernel_size=2, 
            stride=2,
            dim=dim,
        )

        self.final_layer = Conv(
            in_channels=self.final_upsample_dims+in_channels, 
            out_channels=out_channels, 
            kernel_size=1,
            stride=1,
            dim=dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        input = x
        encoded_list = []
        x = self.body.stem(x)
        for downblock in self.body.downblock_layers:
            encoded_list.append(x)
            x = downblock(x)

        for encoded, upblock in zip(reversed(encoded_list), self.upblock_layers):
            x = upblock(x, encoded)

        x = self.final_upsample(x)
        x = torch.cat([input,x], dim=1)
        x = self.final_layer(x)
        # activation?
        return x

    def macs(self) -> float:
        return residualunet_macs(
            dim=self.dim,
            growth_factor=self.body.growth_factor,
            kernel_size=self.body.kernel_size,
            stub_kernel_size=self.body.stub_kernel_size,
            initial_features=self.body.initial_features,
            downblock_layers=self.body.layers,
        )

class SimpleCNN(nn.Module):
    def __init__(
        self,
        dim:int,
        in_channels:int = 1,
        hidden_features:int = 64,
        out_channels:int = 1,
        kernel_size:int = 3,
        layers_num:int = 5
    ):
        super().__init__()

        self.block_layers = [] 
        current_channels = in_channels
        for idx in range(layers_num):
            self.block_layers.append(
                Conv(
                    in_channels=current_channels,
                    out_channels=hidden_features if idx != layers_num-1 else out_channels,
                    kernel_size = kernel_size,
                    padding = "same",
                    dim=dim
                )
            )
            self.block_layers.append(nn.ReLU(inplace=True))
            current_channels = hidden_features

        self.block_layers = nn.Sequential(*self.block_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block_layers(x)

def residualunet_macs(
    dim:int,
    growth_factor:float,
    kernel_size:int,
    stub_kernel_size:int,
    initial_features:int,
    downblock_layers:int,
) -> float:
    """
    M = L + \sum_{i=0}^n D_i + U_i
    D_0 = \frac{\kappa ^ d}{2^d}  f
    D_i = \frac{1}{2^{d(i+1)}} g^{2i-1} ( k^d( 3g + 1 ) + 1) f^2 
    U_i = \frac{f^2}{2^{d i}}  (2^d g^{2i-1} + 2 k^d g^{2i-2}) 
    U_0 = 2^{d-1} f ^ 2
    L = \frac{f}{2} + 1
    """
    stride = 2
    U_0 = 2 ** (dim - 1) * initial_features **2
    L = initial_features/2 + 1
    body_macs = resnetbody_macs(
        dim=dim,
        growth_factor=growth_factor,
        kernel_size=kernel_size,
        stub_kernel_size=stub_kernel_size,
        initial_features=initial_features,
        downblock_layers=downblock_layers,
    )
    M = L + body_macs + U_0

    for i in range(1, downblock_layers+1):
        U_i = initial_features**2/(stride**(dim * i)) * (
            2**dim * growth_factor ** (2 * i - 1) + 
            2 * kernel_size**dim * growth_factor ** ( 2 * i - 2 )
        )

        M += U_i

    return M


def resnetbody_macs(
    dim:int,
    growth_factor:float,
    kernel_size:int,
    stub_kernel_size:int,
    initial_features:int,
    downblock_layers:int,
) -> float:
    """
    M = \sum_{i=0}^n D_i
    D_0 = \frac{\kappa ^ d}{2^d}  f
    D_i = \frac{1}{2^{d(i+1)}} g^{2i-1} ( k^d( 3g + 1 ) + 1) f^2 
    """
    stride = 2
    D_0 = stub_kernel_size **dim * initial_features / (stride ** dim) 
    M = D_0

    for i in range(1, downblock_layers+1):
        D_i = initial_features**2 /(stride**(dim * (i+1))) * growth_factor ** (2 * i - 1) * (
            kernel_size**dim * (3 * growth_factor + 1) + 1
        )
        M += D_i

    return M


def calc_initial_features_residualunet(
    macc:int,
    dim:int,
    growth_factor:float,
    kernel_size:int,
    stub_kernel_size:int,
    downblock_layers:int,
) -> int:
    """
    """
    stride = 2    
    a = 2 ** (dim - 1)
    for i in range(1, downblock_layers+1):
        D_i_over_f2 = 1 /(stride**(dim * (i+1))) * growth_factor ** (2 * i - 1) * (
            kernel_size**dim * (3 * growth_factor + 1) + 1
        )
        U_i_over_f2 = 1/(stride**(dim * i)) * (
            2**dim * growth_factor ** (2 * i - 1) + 
            2 * kernel_size**dim * growth_factor ** ( 2 * i - 2 )
        )

        a += D_i_over_f2 + U_i_over_f2

    b = stub_kernel_size **dim / (stride ** dim) + 0.5
    c = -macc + 1

    initial_features = (-b + np.sqrt(b**2 - 4*a*c))/(2 * a)

    return int(initial_features + 0.5)    
    
