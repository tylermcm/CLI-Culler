"""The ported MANIQA engine runs end to end (random weights: no model is downloaded). Skipped without torch/timm."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("einops")


def test_maniqa_engine_loads_a_checkpoint_and_scores_an_image(tmp_path):
    from PIL import Image

    from aiculler.models.app.maniqa_net import MANIQA
    from aiculler.models.engines import MANIQA_CHECKPOINT, ManiqaEngine

    torch.manual_seed(0)
    net = MANIQA(embed_dim=768, num_outputs=1, dim_mlp=768, patch_size=8, img_size=224, window_size=4, depths=[2, 2],
                 num_heads=[4, 4], num_tab=2, scale=0.8)
    torch.save(net.state_dict(), tmp_path / MANIQA_CHECKPOINT)
    engine = ManiqaEngine(tmp_path, "cpu")
    assert engine.device == "cpu" and not engine.half
    image = Image.new("RGB", (400, 300), (120, 90, 60))
    assert isinstance(engine.score(image), float)
    tiny = engine.score(Image.new("RGB", (100, 80), (120, 90, 60)))  # smaller than a crop: enlarged, still scored
    assert isinstance(tiny, float)


def test_maniqa_engine_refuses_the_wrong_checkpoint(tmp_path):
    from aiculler.models.engines import MANIQA_CHECKPOINT, ManiqaEngine

    torch.save({"unrelated.weight": torch.zeros(1)}, tmp_path / MANIQA_CHECKPOINT)
    with pytest.raises(RuntimeError, match="does not match the MANIQA network"):
        ManiqaEngine(tmp_path, "cpu")


def test_float16_conv3d_runs_in_float32():
    import torch

    from aiculler.models.engines import _conv3d_in_float32

    net = torch.nn.Sequential(torch.nn.Conv3d(3, 4, kernel_size=(2, 4, 4), stride=(2, 4, 4)), torch.nn.Flatten()).half()
    _conv3d_in_float32(torch, net)
    assert net[0].weight.dtype == torch.float32
    out = net(torch.randn(1, 3, 2, 8, 8, dtype=torch.float16))
    assert out.dtype == torch.float16
