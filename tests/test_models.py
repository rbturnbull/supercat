from supercat.models import ResNet, ResidualUNet
import torch

def test_resnet3d():
    res = 100
    samples = 4
    num_classes = 9
    in_channels = 1
    dim = 3
    model = ResNet(in_channels=in_channels, num_classes=num_classes, dim=dim)
    x = torch.rand((samples, in_channels, res, res, res))
    y = model(x)
    assert y.size() == (samples, num_classes)


def test_resnet3d_position_emb():
    res = 100
    position_emb_dim = res // 2
    samples = 4
    num_classes = 9
    in_channels = 1
    dim = 3
    model_emb = ResNet(
        dim=dim,
        in_channels=in_channels,
        num_classes=num_classes,
        position_emb_dim=position_emb_dim,
    )

    model_affine = ResNet(
        dim=dim,
        in_channels=in_channels,
        num_classes=num_classes,
        position_emb_dim=position_emb_dim,
        use_affine=True,
    )

    summary_emb = str(model_emb)
    summary_affine = str(model_affine)

    x = torch.rand((samples, in_channels, res, res, res))
    t = torch.rand((samples, 1))

    y_emb = model_emb(x)
    y_affine = model_affine(x)

    y_emb_t = model_emb(x, t)
    y_affine_t = model_affine(x, t)

    assert "FeatureWiseAffine" in summary_emb
    assert "FeatureWiseAffine" in summary_affine

    assert model_emb.position_encoder.embedding_dim == position_emb_dim
    assert model_affine.position_encoder.embedding_dim == position_emb_dim

    assert y_emb.size() == (samples, num_classes)
    assert y_affine.size() == (samples, num_classes)
    assert y_emb_t.size() == (samples, num_classes)
    assert y_affine_t.size() == (samples, num_classes)


def test_unet3d():
    """
    ResidualUnet
    - no position embedding
    - no self attention layer
    """
    res = 64
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 3
    model = ResidualUNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        attn_layers=(),
    )
    x = torch.rand((samples, in_channels, res, res, res))
    y = model(x)
    assert y.size() == (samples, out_channels, res, res, res)


def test_unet3d_position_emb():
    res = 64
    position_emb_dim = res // 2
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 3
    model_emb = ResidualUNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        downblock_layers=4,
        attn_layers=(),
        position_emb_dim=position_emb_dim,
    )
    model_affine = ResidualUNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        downblock_layers=4,
        attn_layers=(),
        position_emb_dim=position_emb_dim,
        use_affine=True,
    )
    summary_emb = str(model_emb)
    summary_affine = str(model_affine)

    x = torch.rand((samples, in_channels, res, res, res))
    t = torch.rand((samples, 1))

    y_emb = model_emb(x)
    y_affine = model_affine(x)
    y_emb_t = model_emb(x, t)
    y_affine_t = model_affine(x, t)

    assert "FeatureWiseAffine" in summary_emb
    assert "FeatureWiseAffine" in summary_affine

    assert model_emb.position_encoder.embedding_dim == position_emb_dim
    assert model_affine.position_encoder.embedding_dim == position_emb_dim

    assert y_emb.size() == (samples, out_channels, res, res, res)
    assert y_affine.size() == (samples, out_channels, res, res, res)
    assert y_emb_t.size() == (samples, out_channels, res, res, res)
    assert y_affine_t.size() == (samples, out_channels, res, res, res)


def test_unet3d_attn():
    res = 64
    downblock_layers = 4
    attn_layers = (3,)
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 3
    model = ResidualUNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        downblock_layers=downblock_layers,
        attn_layers=attn_layers,
    )
    summary = str(model)
    x = torch.rand((samples, in_channels, res, res, res))
    y = model(x)

    assert "SelfAttention" in summary
    assert model.attn_layers == attn_layers

    for id in range(downblock_layers):
        down_id = id
        up_id = downblock_layers - 1 - id
        assert model.body.downblock_layers[id].use_attn == (down_id in attn_layers)
        assert model.body.downblock_layers[id].block1.use_attn == (
            down_id in attn_layers
        )
        assert model.body.downblock_layers[id].block2.use_attn == (
            down_id in attn_layers
        )
        assert model.upblock_layers[id].block1.use_attn == (up_id in attn_layers)
    assert y.size() == (samples, out_channels, res, res, res)


def test_unet3d_attn_position_emb_affine():
    res = 64
    position_emb_dim = res // 2
    downblock_layers = 4
    attn_layers = (3,)
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 3
    model = ResidualUNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        downblock_layers=downblock_layers,
        attn_layers=attn_layers,
        position_emb_dim=position_emb_dim,
        use_affine=True,
    )

    summary = str(model)
    x = torch.rand((samples, in_channels, res, res, res))
    t = torch.rand((samples, 1))

    y = model(x)
    y_t = model(x, t)

    assert "FeatureWiseAffine" in summary
    assert model.position_encoder.embedding_dim == position_emb_dim

    assert "SelfAttention" in summary
    assert model.attn_layers == attn_layers

    for id in range(downblock_layers):
        down_id = id
        up_id = downblock_layers - 1 - id
        assert model.body.downblock_layers[id].use_attn == (down_id in attn_layers)
        assert model.body.downblock_layers[id].block1.use_attn == (
            down_id in attn_layers
        )
        assert model.body.downblock_layers[id].block2.use_attn == (
            down_id in attn_layers
        )
        assert model.upblock_layers[id].block1.use_attn == (up_id in attn_layers)
    assert y.size() == (samples, out_channels, res, res, res)
    assert y_t.size() == (samples, out_channels, res, res, res)


def test_resnet2d():
    res = 100
    samples = 4
    num_classes = 9
    in_channels = 1
    dim = 2
    model = ResNet(
        dim=dim,
        in_channels=in_channels,
        num_classes=num_classes,
    )
    x = torch.rand((samples, in_channels, res, res))
    y = model(x)
    assert y.size() == (samples, num_classes)


def test_resnet2d_position_emb():
    res = 100
    position_emb_dim = res // 2
    samples = 4
    num_classes = 9
    in_channels = 1
    dim = 2
    model_emb = ResNet(
        dim=dim,
        in_channels=in_channels,
        num_classes=num_classes,
        position_emb_dim=position_emb_dim,
    )
    model_affine = ResNet(
        dim=dim,
        in_channels=in_channels,
        num_classes=num_classes,
        position_emb_dim=position_emb_dim,
        use_affine=True,
    )

    summary_emb = str(model_emb)
    summary_affine = str(model_affine)

    x = torch.rand((samples, in_channels, res, res))
    t = torch.rand((samples, 1))

    y_emb = model_emb(x)
    y_affine = model_affine(x)

    y_emb_t = model_emb(x, t)
    y_affine_t = model_affine(x, t)

    assert "FeatureWiseAffine" in summary_emb
    assert "FeatureWiseAffine" in summary_affine

    assert model_emb.position_encoder.embedding_dim == position_emb_dim
    assert model_affine.position_encoder.embedding_dim == position_emb_dim

    assert y_emb.size() == (samples, num_classes)
    assert y_emb_t.size() == (samples, num_classes)
    assert y_affine.size() == (samples, num_classes)
    assert y_affine_t.size() == (samples, num_classes)


def test_unet2d():
    res = 64
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 2
    model = ResidualUNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        attn_layers=(),
    )
    x = torch.rand((samples, in_channels, res, res))
    y = model(x)
    assert y.size() == (samples, out_channels, res, res)


def test_unet2d_position_emb():
    res = 64
    position_emb_dim = res // 2
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 2
    model_emb = ResidualUNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        downblock_layers=4,
        attn_layers=(),
        position_emb_dim=position_emb_dim,
    )
    model_affine = ResidualUNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        downblock_layers=4,
        attn_layers=(),
        position_emb_dim=position_emb_dim,
        use_affine=True,
    )

    summary_emb = str(model_emb)
    summary_affine = str(model_affine)

    x = torch.rand((samples, in_channels, res, res))
    t = torch.rand((samples, 1))

    y_emb = model_emb(x)
    y_affine = model_affine(x)
    y_emb_t = model_emb(x, t)
    y_affine_t = model_affine(x, t)

    assert "FeatureWiseAffine" in summary_emb
    assert "FeatureWiseAffine" in summary_affine

    assert model_emb.position_encoder.embedding_dim == position_emb_dim
    assert model_affine.position_encoder.embedding_dim == position_emb_dim

    assert y_emb.size() == (samples, out_channels, res, res)
    assert y_affine.size() == (samples, out_channels, res, res)
    assert y_emb_t.size() == (samples, out_channels, res, res)
    assert y_affine_t.size() == (samples, out_channels, res, res)


def test_unet2d_attn():
    res = 64
    downblock_layers = 4
    attn_layers = (3,)
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 2
    model = ResidualUNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        downblock_layers=downblock_layers,
        attn_layers=attn_layers,
    )
    summary = str(model)
    x = torch.rand((samples, in_channels, res, res))
    y = model(x)

    assert "SelfAttention" in summary
    assert model.attn_layers == attn_layers

    for id in range(downblock_layers):
        down_id = id
        up_id = downblock_layers - 1 - id
        assert model.body.downblock_layers[id].use_attn == (down_id in attn_layers)
        assert model.body.downblock_layers[id].block1.use_attn == (
            down_id in attn_layers
        )
        assert model.body.downblock_layers[id].block2.use_attn == (
            down_id in attn_layers
        )
        assert model.upblock_layers[id].block1.use_attn == (up_id in attn_layers)
    assert y.size() == (samples, out_channels, res, res)


def test_unet2d_attn_position_emb_affine():
    res = 64
    position_emb_dim = res // 2
    downblock_layers = 4
    attn_layers = (3,)
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 2
    model = ResidualUNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        downblock_layers=downblock_layers,
        attn_layers=attn_layers,
        position_emb_dim=position_emb_dim,
        use_affine=True,
    )
    summary = str(model)
    x = torch.rand((samples, in_channels, res, res))
    t = torch.rand((samples, 1))

    y = model(x)
    y_t = model(x, t)
    assert "FeatureWiseAffine" in summary
    assert model.position_encoder.embedding_dim == position_emb_dim

    assert "SelfAttention" in summary
    assert model.attn_layers == attn_layers

    for id in range(downblock_layers):
        down_id = id
        up_id = downblock_layers - 1 - id
        assert model.body.downblock_layers[id].use_attn == (down_id in attn_layers)
        assert model.body.downblock_layers[id].block1.use_attn == (
            down_id in attn_layers
        )
        assert model.body.downblock_layers[id].block2.use_attn == (
            down_id in attn_layers
        )
        assert model.upblock_layers[id].block1.use_attn == (up_id in attn_layers)
    assert y.size() == (samples, out_channels, res, res)
    assert y_t.size() == (samples, out_channels, res, res)


def test_unet2d_growth_factor():
    res = 64
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 2
    model = ResidualUNet(
        dim=dim, in_channels=in_channels, out_channels=out_channels, growth_factor=1.5
    )
    summary = str(model)
    assert "(conv1): Conv2d(64, 96," in summary
    x = torch.rand((samples, in_channels, res, res))
    y = model(x)
    assert y.size() == (samples, out_channels, res, res)


def test_unet2d_growth_factor_attn_position_emb():
    res = 64
    position_emb_dim = res // 2
    downblock_layers = 4
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 2
    model = ResidualUNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        growth_factor=1.5,
        downblock_layers=downblock_layers,
        attn_layers=(3,),
        position_emb_dim=position_emb_dim,
        use_affine=True,
    )

    summary = str(model)
    assert "(conv1): Conv2d(64, 96," in summary
    x = torch.rand((samples, in_channels, res, res))
    y = model(x)
    assert y.size() == (samples, out_channels, res, res)


def test_unet3d_growth_factor():
    res = 50
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 3
    model = ResidualUNet(
        dim=dim,
        in_channels=in_channels,
        initial_features=17,
        out_channels=out_channels,
        growth_factor=1.5,
    )
    summary = str(model)
    assert "(conv1): Conv3d(17, 25," in summary
    x = torch.rand((samples, in_channels, res, res, res))
    y = model(x)
    assert y.size() == (samples, out_channels, res, res, res)


def test_unet3d_growth_factor_attn_position_emb():
    res = 50
    position_emb_dim = res // 2
    downblock_layers = 4
    samples = 2
    in_channels = 1
    out_channels = 2
    dim = 3
    model = ResidualUNet(
        dim=dim,
        in_channels=in_channels,
        initial_features=17,
        out_channels=out_channels,
        growth_factor=1.5,
        downblock_layers=downblock_layers,
        attn_layers=(3,),
        position_emb_dim=position_emb_dim,
        use_affine=True,
    )
    summary = str(model)
    assert "(conv1): Conv3d(17, 25," in summary
    x = torch.rand((samples, in_channels, res, res, res))
    y = model(x)
    assert y.size() == (samples, out_channels, res, res, res)


def test_unet3d_res100():
    res = 100
    samples = 2
    in_channels = 1
    out_channels = 3
    dim = 3
    model = ResidualUNet(in_channels=in_channels, out_channels=out_channels, dim=dim)
    x = torch.rand((samples, in_channels, res, res, res))
    y = model(x)
    assert y.size() == (samples, out_channels, res, res, res)


def test_unet3d_res100_attn_position_emb():
    res = 100
    position_emb_dim = res // 2
    downblock_layers = 4
    samples = 2
    in_channels = 1
    out_channels = 3
    dim = 3
    model = ResidualUNet(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        downblock_layers=downblock_layers,
        attn_layers=(3,),
        position_emb_dim=position_emb_dim,
        use_affine=True,
    )
    x = torch.rand((samples, in_channels, res, res, res))
    y = model(x)
    assert y.size() == (samples, out_channels, res, res, res)
