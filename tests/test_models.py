from supercat.models import ResNet3d, Unet3d
import torch

def test_resnet3d():
    res = 100
    samples = 4
    num_classes = 9
    in_channels = 1
    model = ResNet3d(in_channels=in_channels, num_classes=num_classes)
    x = torch.rand( (samples, in_channels, res,res,res) )
    y = model(x)
    assert y.size() == (samples, num_classes)

def test_unet3d():
    res = 64
    samples = 2
    in_channels = 1
    out_channels = 2
    model = Unet3d(in_channels=in_channels, out_channels=out_channels)
    x = torch.rand( (samples, in_channels, res,res,res) )
    y = model(x)
    assert y.size() == (samples, out_channels, res, res, res)


def test_unet3d_res100():
    res = 100
    samples = 2
    in_channels = 1
    out_channels = 3
    model = Unet3d(in_channels=in_channels, out_channels=out_channels)
    x = torch.rand( (samples, in_channels, res,res,res) )
    y = model(x)
    assert y.size() == (samples, out_channels, res, res, res)

