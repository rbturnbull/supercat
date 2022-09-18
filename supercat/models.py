import torch
from torch import nn
from torch import Tensor
from torchvision.models import video
from torchvision.models.video.resnet import VideoResNet
from fastai.vision.learner import _load_pretrained_weights, _get_first_layer

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

        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, dim=dim)
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


class ResNet(nn.Module):
    def __init__(
        self,
        dim:int,
        num_classes:int=1,
        body = None,
        in_channels:int = 1,
        initial_features:int = 64,
        growth_factor:float = 2.0,
    ):
        super().__init__()
        self.body = body if body is not None else ResNetBody(
            dim=dim, 
            in_channels=in_channels, 
            initial_features=initial_features, 
            growth_factor=growth_factor
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
    ):
        super().__init__()
        self.body = body if body is not None else ResNetBody(
            dim=dim, 
            in_channels=in_channels, 
            initial_features=initial_features, 
            growth_factor=growth_factor,
            kernel_size=kernel_size,
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


class DoNothing(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        # dummy layer so that there are some parameters
        self.dummy = nn.Conv3d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x


def get_in_out_channels(layer):
    first_conv = next(next(next(layer.children()).children()).children())
    return first_conv.in_channels, first_conv.out_channels


def update_first_layer(model, n_in=1, pretrained=True):
    first_layer, parent, name = _get_first_layer(model)
    assert isinstance(first_layer, (nn.Conv2d, nn.Conv3d)), f'Change of input channels only supported with Conv2d or Conv3d, found {first_layer.__class__.__name__}'
    assert getattr(first_layer, 'in_channels') == 3, f'Unexpected number of input channels, found {getattr(first_layer, "in_channels")} while expecting 3'
    params = {attr:getattr(first_layer, attr) for attr in 'out_channels kernel_size stride padding dilation groups padding_mode'.split()}
    params['bias'] = getattr(first_layer, 'bias') is not None
    params['in_channels'] = n_in
    new_layer = type(first_layer)(**params)
    if pretrained:
        _load_pretrained_weights(new_layer, first_layer)
    setattr(parent, name, new_layer)


class VideoUnet3d(nn.Module):
    def __init__(self, base:VideoResNet = None, in_channels:int = 1, out_channels:int = 1, pretrained:bool = True, **kwargs):
        super().__init__(**kwargs)
        base = base or video.r3d_18(pretrained=pretrained)
        self.base = base
        self.initial_in_channels = in_channels
        self.final_out_channels = out_channels
        
        self.base_layers = {name:child for name,child in base.named_children()}

        # Edit the first layer as needed
        if in_channels != 3:
            update_first_layer(base, in_channels, pretrained=pretrained)
        first_layer = next(base.stem.children())
        first_layer.stride = (2,2,2)
        first_layer.padding = (1,3,3)

        in_channels, out_channels = get_in_out_channels(self.base_layers['layer4'])
        self.up_block4 = UpBlock(in_channels=out_channels, out_channels=in_channels)
        in_channels, out_channels = get_in_out_channels(self.base_layers['layer3'])
        self.up_block3 = UpBlock(in_channels=out_channels, out_channels=in_channels)
        in_channels, out_channels = get_in_out_channels(self.base_layers['layer2'])
        self.up_block2 = UpBlock(in_channels=out_channels, out_channels=in_channels)
        in_channels, out_channels = get_in_out_channels(self.base_layers['layer1'])
        self.up_block1 = UpBlock(in_channels=out_channels, out_channels=in_channels)

        self.final_upsample_dims = out_channels//2
        self.final_upsample = nn.ConvTranspose3d(
            in_channels=out_channels, 
            out_channels=self.final_upsample_dims, 
            kernel_size=2, 
            stride=2
        )

        self.final_layer = nn.Conv3d(
            in_channels=self.final_upsample_dims+self.initial_in_channels, 
            out_channels=self.final_out_channels, 
            kernel_size=1,
            stride=1,
        )        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x

        x = encoded_0 = self.base.stem(x)
        x = encoded_1 = self.base.layer1(x)
        x = encoded_2 = self.base.layer2(x)
        x = encoded_3 = self.base.layer3(x)
        x = self.base.layer4(x)

        x = self.up_block4(x, encoded_3)
        x = self.up_block3(x, encoded_2)
        x = self.up_block2(x, encoded_1)
        x = self.up_block1(x, encoded_0)
        x = self.final_upsample(x)
        x = torch.cat([input,x], dim=1)
        x = self.final_layer(x)
        # activation?
        return x    


# out_channels(in_channels * k**d + 1)