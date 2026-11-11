import torch

from dgm.nets import (
    MLP,
    ConvDecoder,
    ConvEncoder,
    TimeConditionedMLP,
    UNet,
    build_decoder,
    build_encoder,
    build_time_conditioned_net,
)


def test_mlp_shapes():
    m = MLP((2,), 3, hidden=16, n_layers=2)
    assert m(torch.randn(5, 2)).shape == (5, 3)
    m2 = MLP(4, (2, 2), hidden=8, n_layers=1)
    assert m2(torch.randn(5, 4)).shape == (5, 2, 2)


def test_time_mlp():
    net = TimeConditionedMLP((2,), hidden=32, n_layers=2, time_dim=16)
    x, t = torch.randn(7, 2), torch.rand(7)
    assert net(x, t).shape == (7, 2)
    assert net(x, torch.tensor(0.5)).shape == (7, 2)


def test_unet_shape_small():
    net = UNet(
        in_ch=3, ch=8, ch_mult=(1, 2), num_res_blocks=1, attn_resolutions=(8,), resolution=16
    )
    x, t = torch.randn(2, 3, 16, 16), torch.rand(2)
    y = net(x, t)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_conv_enc_dec():
    enc = ConvEncoder(3, 10, ch=8, resolution=32)
    dec = ConvDecoder(10, 3, ch=8, resolution=32)
    z = enc(torch.randn(4, 3, 32, 32))
    assert z.shape == (4, 10)
    x = dec(z)
    assert x.shape == (4, 3, 32, 32) and x.abs().max() <= 1.0


def test_factory_dispatch():
    cfg = {
        "mlp": {"hidden": 16, "n_layers": 1, "time_dim": 8},
        "unet": {"ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": [8]},
        "conv": {"ch": 8},
    }
    assert isinstance(build_time_conditioned_net((2,), cfg), TimeConditionedMLP)
    assert isinstance(build_time_conditioned_net((3, 16, 16), cfg), UNet)
    assert isinstance(build_encoder((2,), 4, cfg), MLP)
    assert isinstance(build_encoder((3, 32, 32), 4, cfg), ConvEncoder)
    assert isinstance(build_decoder((2,), 4, cfg), MLP)
    assert isinstance(build_decoder((3, 32, 32), 4, cfg, out_act="tanh"), ConvDecoder)
    assert build_decoder((2,), 4, cfg)(torch.randn(3, 4)).shape == (3, 2)
