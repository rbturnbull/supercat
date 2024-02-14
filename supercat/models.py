import torch
from torch import nn
from torch import Tensor
import numpy as np
import math
from supercat.enums import PaddingMode

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


def Conv(*args, dim:int, padding_mode:str='reflect', **kwargs):
    if dim == 2:
        return nn.Conv2d(padding_mode=padding_mode, *args, **kwargs)
    if dim == 3:
        return nn.Conv3d(padding_mode=padding_mode, *args, **kwargs)
    raise ValueError(f"dimension {dim} not supported")    
    

def BatchNorm(*args, dim:int, **kwargs):
    return nn.Identity()
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
    

class PositionalEncoding(nn.Module):
    """
    Transforming time/noise values into embedding vectors.

    Taken from: https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/sr3_modules/unet.py#L18
    """

    def __init__(self, embedding_dim):
        """
        Arguments:
            embedding_dim:
                The dimension of the output positional embedding
        """
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, position_level):
        """
        Arguments:
            position_level:
                The positional information to be encoded.  Can be either time value or noise variance value.
        """
        count = self.embedding_dim // 2
        step = torch.arange(count, dtype=position_level.dtype, device=position_level.device) / count

        encoding = position_level.unsqueeze(1) * torch.exp(- math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)

        return encoding

class FeatureWiseAffine(nn.Module):
    """
    FiLM layer that integrage noise/time information into the input image
    
    Taken from: https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/sr3_modules/unet.py#L34
    Based on: https://distill.pub/2018/feature-wise-transformations/
    """

    def __init__(self, dim: int, embedding_dim: int, image_channels: int, use_affine: bool=False):
        """
        Arguments: 
            dim:
                the dimension of the image. Value should be 2 or 3
            embedding_dim:
                the length of the noise embedding
            image_channels:
                the length of the noise embedding. This value will equal to the input image's channel size
            use_affine:
                Whether to use FeatureWiseAffine to integrate the noise information.  If False, the noise_emb will
                simply be projected and reshape to match the image size, then added to the image.
        """
        super(FeatureWiseAffine, self).__init__()
        self.dim = dim # dimension of the image: 2D or 3D
        self.use_affine = use_affine
        self.noise_func = nn.Sequential(
            nn.Linear(embedding_dim, image_channels * (1 + use_affine))
        )

    def forward(self, x, position_emb):
        """
        Arguments:
            x:
                the target image that the function is altering
            position_emb:
                the vector representation of the position level information.
        Return:
            x:
                the image altered based on the position level information
        """
        batch = x.shape[0]

        if self.dim == 2:
            position_emb = self.noise_func(position_emb).view(batch, -1, 1, 1)
        elif self.dim == 3:
            position_emb = self.noise_func(position_emb).view(batch, -1, 1, 1, 1)

        if self.use_affine:
            gamma, beta = position_emb.chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + position_emb

        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, in_channels, num_heads:int=1, padding_mode:str=PaddingMode.REFLECT.value) -> None:
        """
        Arguments:
            dim:
                the dimension of the image. Value should be 2 or 3
            in_channels:
                the number of channel of the image the module is self-attented to
            num_heads:
                the number of heads used in the self attntion module
        """
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.norm = BatchNorm(in_channels, dim=dim)
        self.qkv_generator = Conv(in_channels, in_channels * 3, kernel_size=1, stride =1, bias=False, padding_mode=padding_mode, dim=dim)
        self.output = Conv(in_channels, in_channels, kernel_size=1, padding_mode=padding_mode, dim=dim)

        if dim == 2:
            self.attn_mask_eq = "bnchw, bncyx -> bnhwyx"
            self.attn_value_eq = "bnhwyx, bncyx -> bnchw"
        elif dim == 3:
            self.attn_mask_eq = "bncdhw, bnczyx -> bndhwzyx"
            self.attn_value_eq = "bndhwzyx, bnczyx -> bncdhw"


    def forward(self, x):

        head_dim = x.shape[1] // self.num_heads

        normalised_x = self.norm(x)

        # compute query key value vectors
        qkv = self.qkv_generator(normalised_x).view(x.shape[0], self.num_heads, head_dim * 3, *x.shape[2:])
        query, key, value = qkv.chunk(3, dim=2) # split qkv along the head_dim axis

        # compute attention mask
        attn_mask = torch.einsum(self.attn_mask_eq, query, key) / math.sqrt(x.shape[1])
        attn_mask = attn_mask.view(x.shape[0], self.num_heads, *x.shape[2:], -1)
        attn_mask = torch.softmax(attn_mask, -1)
        attn_mask = attn_mask.view(x.shape[0], self.num_heads, *x.shape[2:], *x.shape[2:])

        #compute attntion value
        attn_value = torch.einsum(self.attn_value_eq, attn_mask, value)
        attn_value = attn_value.view(*x.shape)

        return x + self.output(attn_value)


class ResBlock(nn.Module):
    """ 
    Based on
        https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448 
        https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        downsample: bool,
        padding_mode: str = PaddingMode.REFLECT.value,
        kernel_size: int = 3,
        position_emb_dim: int = None,
        use_affine: bool = False,
        use_attn: bool=False
    ):
        super().__init__()
        self.padding_mode = padding_mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.affine = use_affine
        self.use_attn = use_attn

        # calculate padding so that the output is the same as a kernel size of 1 with zero padding
        # this is required to be calculated becaues padding="same" doesn't work with a stride
        padding = (kernel_size - 1)//2

        # position_emb_dim is used as an idicator for incorporating position information or not
        self.position_emb_dim = position_emb_dim
        if position_emb_dim is not None:
            self.noise_func = FeatureWiseAffine(
                dim=dim,
                embedding_dim=position_emb_dim,
                image_channels=out_channels,
                use_affine=use_affine
            )

        if downsample:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, padding_mode=padding_mode, dim=dim)
            self.shortcut = nn.Sequential(
                Conv(in_channels, out_channels, kernel_size=1, stride=2, padding_mode=padding_mode, dim=dim), 
                BatchNorm(out_channels, dim=dim)
            )
        else:
            self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode=padding_mode, dim=dim)
            self.shortcut = nn.Sequential()

        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode=padding_mode, dim=dim)
        self.bn1 = BatchNorm(out_channels, dim=dim)
        self.bn2 = BatchNorm(out_channels, dim=dim)
        self.relu = nn.ReLU(inplace=True)

        if use_attn:
            self.attn = SelfAttention(dim=dim, in_channels=out_channels, padding_mode=padding_mode)

    def forward(self, x: Tensor, position_emb: Tensor = None):
        input = x
        shortcut = self.shortcut(x)
        # print('shortcut max', shortcut.max())
        x = self.relu(self.bn1(self.conv1(x)))

        # print('block 1 max', x.max())
        # incorporate position information only if position_emb is provided and noise_func exist
        if position_emb is not None and self.position_emb_dim is not None:
            x  = self.noise_func (x, position_emb)

        x = self.relu(self.bn2(self.conv2(x)))
        # print('block 2 max', x.max())

        x = self.relu(x + shortcut)

        if self.use_attn:
            x = self.attn(x)

        # if not torch.isfinite(x).all():
        #     breakpoint()

        return x


class DownBlock(nn.Module):
    def __init__(
        self,
        dim:int,
        padding_mode: str = PaddingMode.REFLECT.value,
        in_channels:int = 1,
        downsample:bool = True,
        growth_factor:float = 2.0,
        kernel_size:int = 3,
        position_emb_dim:int = None,
        use_affine:bool = False,
        use_attn:bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.position_emb_dim = position_emb_dim
        self.use_affine = use_affine
        self.use_attn = use_attn
        self.padding_mode = padding_mode

        if downsample:
            self.out_channels = int(growth_factor*self.out_channels)

        self.block1 = ResBlock(
            dim=dim,
            padding_mode=padding_mode,
            in_channels=in_channels,
            out_channels=self.out_channels,
            downsample=downsample,
            kernel_size=kernel_size,
            position_emb_dim=position_emb_dim,
            use_affine=use_affine,
            use_attn=use_attn,
        )
        self.block2 = ResBlock(
            dim=dim,
            padding_mode=padding_mode,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            downsample=False,
            kernel_size=kernel_size,
            position_emb_dim=position_emb_dim,
            use_affine=use_affine,
            use_attn=use_attn,
        )

    def forward(self, x: Tensor, position_emb: Tensor = None) -> Tensor:
        x1 = self.block1(x, position_emb)
        # if not torch.isfinite(x1).all():
        #     breakpoint()
        x2 = self.block2(x1, position_emb)
        # if not torch.isfinite(x2).all():
        #     breakpoint()
        return x2


class UpBlock(nn.Module):
    def __init__(
        self,
        dim:int,
        in_channels:int,
        out_channels:int, 
        padding_mode:str = PaddingMode.REFLECT.value,
        resblock_kernel_size:int = 3,
        upsample_kernel_size:int = 2,
        position_emb_dim: int = None,
        use_affine: bool = False,
        use_attn: bool = False
    ):
        super().__init__()
        self.padding_mode = padding_mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.position_emb_dim = position_emb_dim
        self.use_affine = use_affine
        self.use_attn = use_attn

        self.upsample = ConvTranspose(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=upsample_kernel_size,
            stride=2,
            dim=dim
        )

        self.block1 = ResBlock(
            dim=dim,
            padding_mode=padding_mode,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            downsample=False,
            kernel_size=resblock_kernel_size,
            position_emb_dim=position_emb_dim,
            use_affine=use_affine,
            use_attn=use_attn
        )
        # self.block2 = ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, downsample=False, dim=dim, kernel_size=resblock_kernel_size)

    def forward(self, x: Tensor, shortcut: Tensor, position_emb: Tensor = None) -> Tensor:
        x = self.upsample(x)
        # crop upsampled tensor in case the size is different from the shortcut connection
        x, shortcut = autocrop(x, shortcut)
        """ should be concatenation, is there a reason for this implementation """
        x += shortcut
        x = self.block1(x, position_emb)
        # x = self.block2(x)
        return x


class ResNetBody(nn.Module):
    def __init__(
        self,
        dim:int,
        padding_mode: str = PaddingMode.REFLECT.value,
        in_channels:int = 1,
        initial_features:int = 64,
        growth_factor:float = 2.0,
        kernel_size:int = 3,
        stub_kernel_size:int = 7,
        layers:int = 4,
        attn_layers=(3,),
        position_emb_dim:int = None,
        use_affine:bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.padding_mode = padding_mode
        self.in_channels = in_channels
        self.initial_features = initial_features
        self.growth_factor = growth_factor
        self.kernel_size = kernel_size
        self.stub_kernel_size = stub_kernel_size
        self.layers = layers
        self.attn_layers = attn_layers
        self.position_emb_dim = position_emb_dim
        self.use_affine = use_affine

        current_num_features = initial_features
        padding = (stub_kernel_size - 1)//2

        self.stem = nn.Sequential(
            Conv(in_channels=in_channels, out_channels=current_num_features, kernel_size=stub_kernel_size, stride=2, padding=padding, padding_mode=padding_mode, dim=dim),
            BatchNorm(num_features=current_num_features, dim=dim),
            nn.ReLU(inplace=True),
        )

        self.downblock_layers = nn.ModuleList()
        for layer_idx in range(layers):
            downblock = DownBlock(
                dim=dim, 
                padding_mode=padding_mode,
                in_channels=current_num_features,
                downsample=True,
                growth_factor=growth_factor, 
                kernel_size=kernel_size,
                position_emb_dim=position_emb_dim,
                use_affine=use_affine,
                use_attn = (layer_idx in attn_layers),
            )
            self.downblock_layers.append(downblock)
            current_num_features = downblock.out_channels

        self.output_features = current_num_features

    def forward(self, x: Tensor, position_emb: Tensor = None) -> Tensor:
        x = self.stem(x)
        for layer in self.downblock_layers:
            x = layer(x, position_emb)
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
        padding_mode: str = PaddingMode.REFLECT.value,
        body: ResNetBody = None,
        in_channels:int = 1,
        initial_features:int = 64,
        growth_factor:float = 2.0,
        layers:int = 4,
        attn_layers=(),
        position_emb_dim:int = None,
        use_affine:bool = False,
    ):
        super().__init__()

        self.position_emb_dim = position_emb_dim
        self.padding_mode = padding_mode

        if position_emb_dim is not None:
            self.position_encoder = PositionalEncoding(position_emb_dim)

        self.body = body if body is not None else ResNetBody(
            dim=dim, 
            padding_mode=padding_mode,
            in_channels=in_channels,
            initial_features=initial_features,
            growth_factor=growth_factor,
            layers=layers,
            attn_layers=attn_layers,
            position_emb_dim=position_emb_dim,
            use_affine=use_affine
        )
        assert in_channels == self.body.in_channels
        assert initial_features == self.body.initial_features
        assert growth_factor == self.body.growth_factor
        assert layers == self.body.layers
        assert attn_layers == self.body.attn_layers
        assert position_emb_dim == self.body.position_emb_dim
        assert use_affine == self.body.use_affine

        self.global_average_pool = AdaptiveAvgPool(1, dim=dim)
        self.final_layer = torch.nn.Linear(self.body.output_features, num_classes)

    def forward(self, x: Tensor, position: Tensor = None) -> Tensor:
        if self.position_emb_dim is not None and position is not None:
            position_emb = self.position_encoder(position)
        else:
            position_emb = None

        x = self.body(x, position_emb)

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
        padding_mode: str = PaddingMode.REFLECT.value,
        in_channels:int = 1,
        initial_features:int = 64,
        out_channels: int = 1,
        growth_factor:float = 2.0,
        kernel_size:int = 3,
        downblock_layers:int = 4,
        attn_layers = (3,),
        position_emb_dim:int = None,
        use_affine:bool = False
    ):
        super().__init__()
        self.dim = dim
        self.attn_layers = attn_layers
        self.position_emb_dim = position_emb_dim
        self.use_affine = use_affine
        self.padding_mode = padding_mode

        if position_emb_dim is not None:
            self.position_encoder = PositionalEncoding(position_emb_dim)

        self.body = body if body is not None else ResNetBody(
            dim=dim, 
            padding_mode= padding_mode,
            in_channels=in_channels, 
            initial_features=initial_features, 
            growth_factor=growth_factor,
            kernel_size=kernel_size,
            layers=downblock_layers,
            attn_layers=attn_layers,
            position_emb_dim=position_emb_dim,
            use_affine = use_affine
        )
        assert in_channels == self.body.in_channels
        assert initial_features == self.body.initial_features
        assert growth_factor == self.body.growth_factor
        assert kernel_size == self.body.kernel_size
        assert downblock_layers == self.body.layers
        assert attn_layers == self.body.attn_layers
        assert position_emb_dim == self.body.position_emb_dim
        assert use_affine == self.body.use_affine

        self.upblock_layers = nn.ModuleList()
        for downblock in reversed(self.body.downblock_layers):
            upblock = UpBlock(
                dim=dim, 
                padding_mode=padding_mode,
                in_channels=downblock.out_channels,
                out_channels=downblock.in_channels,
                resblock_kernel_size=kernel_size,
                position_emb_dim=position_emb_dim,
                use_affine=use_affine,
                use_attn=downblock.use_attn
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
            padding_mode=padding_mode,
            in_channels=self.final_upsample_dims+in_channels, 
            out_channels=out_channels, 
            kernel_size=1,
            stride=1,
            dim=dim,
        )

    def forward(self, x: Tensor, position: Tensor = None) -> Tensor:
        if self.position_emb_dim is not None and position is not None:
            position_emb = self.position_encoder(position)
        else:
            position_emb = None

        x = x.float()
        input = x
        encoded_list = []
        x = self.body.stem(x)

        for downblock in self.body.downblock_layers:
            encoded_list.append(x)
            x = downblock(x, position_emb)
            
        for encoded, upblock in zip(reversed(encoded_list), self.upblock_layers):
            x = upblock(x, encoded, position_emb)

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
    
