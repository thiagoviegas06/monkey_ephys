from __future__ import annotations

from pathlib import Path
import tempfile

import torch

from src.model import build_model
from src.predict import _load_model, _state_dict_fingerprint


def _dummy_batch(batch_size: int = 4, window_size: int = 201) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_sbp = torch.randn(batch_size, window_size, 96)
    x_kin = torch.randn(batch_size, window_size, 4)

    mask = torch.zeros(batch_size, 96, dtype=torch.bool)
    mask[:, ::4] = True
    x_sbp[:, :, ::4] = 0.0

    obs_mask = (~mask).to(torch.float32).unsqueeze(1).repeat(1, window_size, 1)
    return x_sbp, x_kin, obs_mask


def test_forward_shape_and_weight_sensitivity() -> None:
    torch.manual_seed(42)
    model_a = build_model("tcn")
    x_sbp, x_kin, obs_mask = _dummy_batch()
    y_a = model_a(x_sbp, x_kin, obs_mask)
    assert y_a.shape == (4, 96, 201), f"Expected (4,96,201), got {tuple(y_a.shape)}"

    torch.manual_seed(43)
    model_b = build_model("tcn")
    y_b = model_b(x_sbp, x_kin, obs_mask)
    max_abs_diff = float((y_a - y_b).abs().max().item())
    assert max_abs_diff > 1e-6, f"Different initializations produced near-identical outputs: {max_abs_diff}"
    print(f"[smoke] forward shape OK; max_abs_diff(model_a, model_b)={max_abs_diff:.8f}")


def test_checkpoint_ab_non_identity() -> None:
    device = torch.device("cpu")

    torch.manual_seed(100)
    model_a = build_model("tcn")
    torch.manual_seed(101)
    model_b = build_model("tcn")

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_a = Path(tmp_dir) / "a.pt"
        ckpt_b = Path(tmp_dir) / "b.pt"
        kwargs = {
            "input_dim": 196,
            "hidden_dim": 192,
            "output_dim": 96,
            "num_layers": 8,
            "kernel_size": 7,
            "dropout": 0.15,
            "dilation_base": 2,
            "dilation_cap": 64,
            "gn_groups": 8,
        }

        torch.save({"model_name": "tcn", "model_kwargs": kwargs, "model_state_dict": model_a.state_dict()}, ckpt_a)
        torch.save({"model_name": "tcn", "model_kwargs": kwargs, "model_state_dict": model_b.state_dict()}, ckpt_b)

        loaded_a, ckpt_obj_a = _load_model(str(ckpt_a), device=device, debug=False, label="A")
        loaded_b, ckpt_obj_b = _load_model(str(ckpt_b), device=device, debug=False, label="B")

        fp_a = _state_dict_fingerprint(ckpt_obj_a["model_state_dict"])
        fp_b = _state_dict_fingerprint(ckpt_obj_b["model_state_dict"])
        assert fp_a != fp_b, "Fingerprints unexpectedly identical for different checkpoints"

        x_sbp, x_kin, obs_mask = _dummy_batch()
        with torch.no_grad():
            y_a = loaded_a(x_sbp, x_kin, obs_mask)
            y_b = loaded_b(x_sbp, x_kin, obs_mask)
        max_abs_diff = float((y_a - y_b).abs().max().item())
        assert max_abs_diff > 1e-6, "A/B smoke test failed: first-batch outputs are identical"

    print(f"[smoke] A/B checkpoint difference OK; max_abs_diff(first_batch_y_hat)={max_abs_diff:.8f}")


def main() -> None:
    test_forward_shape_and_weight_sensitivity()
    test_checkpoint_ab_non_identity()
    print("[smoke] all checks passed")


if __name__ == "__main__":
    main()

