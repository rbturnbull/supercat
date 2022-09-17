from supercat.models import ResNet, ResidualUNet
import torch

def test_resnet3d():
    res = 100
    samples = 4
    num_classes = 9
    in_channels = 1
    dim = 3
    model = ResNet(in_channels=in_channels, num_classes=num_classes, dim=dim)
    x = torch.rand( (samples, in_channels, res,res,res) )
    y = model(x)
    assert y.size() == (samples, num_classes)


def test_unet3d():
    res = 64
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 3
    model = ResidualUNet(in_channels=in_channels, out_channels=out_channels, dim=dim)
    x = torch.rand( (samples, in_channels, res,res,res) )
    y = model(x)
    assert y.size() == (samples, out_channels, res, res, res)


def test_unet2d():
    res = 64
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 2
    model = ResidualUNet(in_channels=in_channels, out_channels=out_channels, dim=dim)
    summary = str(model)
    x = torch.rand( (samples, in_channels, res,res) )
    y = model(x)
    assert y.size() == (samples, out_channels, res, res)


def test_unet2d_growth_factor():
    res = 64
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 2
    model = ResidualUNet(in_channels=in_channels, out_channels=out_channels, dim=dim, growth_factor=1.5)
    summary = str(model)
    assert "(conv1): Conv2d(64, 96," in summary
    x = torch.rand( (samples, in_channels, res,res) )
    y = model(x)
    assert y.size() == (samples, out_channels, res, res)


def test_unet3d_growth_factor():
    res = 50
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 3
    model = ResidualUNet(in_channels=in_channels, out_channels=out_channels, dim=dim, growth_factor=1.5, initial_features=17)
    summary = str(model)
    assert "(conv1): Conv3d(17, 25," in summary
    x = torch.rand( (samples, in_channels, res,res,res) )
    y = model(x)
    assert y.size() == (samples, out_channels, res, res, res)


def test_unet3d_res100():
    res = 100
    samples = 2
    in_channels = 1
    out_channels = 3
    dim = 3
    model = ResidualUNet(in_channels=in_channels, out_channels=out_channels, dim=dim)
    x = torch.rand( (samples, in_channels, res,res,res) )
    y = model(x)
    assert y.size() == (samples, out_channels, res, res, res)

