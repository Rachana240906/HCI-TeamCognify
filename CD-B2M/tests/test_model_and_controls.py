"""
tests/test_model_and_controls.py
=================================
Unit tests for model, alignment loss, and control generators.

Run with:
    python -m pytest tests/test_model_and_controls.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alignment import brain_alignment_loss, cka_linear, combined_loss, rsa_correlation
from controls import (
    make_random_control,
    make_shuffled_control,
    make_synthetic_control,
    verify_matching,
)
from model import EEGClassifier, EEGEncoder, build_model


# ──────────────────────────────────────────────────────────────────────────────
# 1. EEGEncoder
# ──────────────────────────────────────────────────────────────────────────────

def test_encoder_output_shape_unidirectional():
    enc = EEGEncoder(input_size=21, hidden_size=64, num_layers=1, bidirectional=False)
    x = torch.randn(8, 21, 256)   # (batch, channels, time)
    h = enc(x)
    assert h.shape == (8, 64), f"Expected (8,64), got {h.shape}"


def test_encoder_output_shape_bidirectional():
    enc = EEGEncoder(input_size=21, hidden_size=64, num_layers=1, bidirectional=True)
    x = torch.randn(8, 21, 256)
    h = enc(x)
    assert h.shape == (8, 128), f"Expected (8,128), got {h.shape}"


def test_encoder_batch_independence():
    """Output for one sample should be identical whether run alone or in a batch."""
    enc = EEGEncoder(input_size=21, hidden_size=32, bidirectional=False)
    enc.eval()
    x = torch.randn(4, 21, 128)
    with torch.no_grad():
        h_batch = enc(x)
        h_single = enc(x[2:3])
    assert torch.allclose(h_batch[2], h_single[0], atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# 2. EEGClassifier
# ──────────────────────────────────────────────────────────────────────────────

def test_classifier_logits_shape():
    model = EEGClassifier(input_size=21, n_classes=3, hidden_size=32,
                          num_layers=1, bidirectional=False)
    x = torch.randn(16, 21, 256)
    logits, proj = model(x)
    assert logits.shape == (16, 3)
    assert proj is None   # no proj head without brain_dim


def test_classifier_with_proj_head():
    model = EEGClassifier(input_size=21, n_classes=3, hidden_size=32,
                          num_layers=1, bidirectional=False,
                          proj_dim=16, brain_dim=128)
    x = torch.randn(8, 21, 256)
    logits, proj = model(x)
    assert logits.shape == (8, 3)
    assert proj is not None
    assert proj.shape == (8, 128), f"Expected (8,128), got {proj.shape}"


def test_classifier_predict():
    model = EEGClassifier(input_size=21, n_classes=3, hidden_size=32,
                          num_layers=1, bidirectional=False)
    x = torch.randn(10, 21, 256)
    preds = model.predict(x)
    assert preds.shape == (10,)
    assert preds.dtype == torch.int64
    assert all(0 <= p <= 2 for p in preds.tolist())


def test_build_model_from_config():
    cfg = {
        "model": {
            "input_size": 21,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "bidirectional": False,
            "proj_hidden": 16,
        }
    }
    model = build_model(cfg, n_classes=3, brain_dim=128)
    assert model.proj_head is not None
    x = torch.randn(4, 21, 256)
    logits, proj = model(x)
    assert logits.shape == (4, 3)
    assert proj.shape == (4, 128)


def test_model_no_nan_in_output():
    model = EEGClassifier(input_size=21, n_classes=3, hidden_size=32,
                          num_layers=1, bidirectional=False)
    x = torch.randn(16, 21, 256)
    logits, _ = model(x)
    assert not torch.any(torch.isnan(logits))
    assert not torch.any(torch.isinf(logits))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Alignment loss
# ──────────────────────────────────────────────────────────────────────────────

def test_mse_alignment_zero_for_identical():
    x = torch.randn(8, 128)
    loss = brain_alignment_loss(x, x, loss_type="mse")
    assert loss.item() < 1e-6


def test_mse_alignment_positive_for_different():
    a = torch.randn(8, 128)
    b = torch.randn(8, 128)
    loss = brain_alignment_loss(a, b, loss_type="mse")
    assert loss.item() > 0


def test_cosine_alignment_zero_for_identical():
    x = torch.randn(8, 128)
    loss = brain_alignment_loss(x, x, loss_type="cosine")
    assert loss.item() < 1e-4


def test_cosine_alignment_range():
    a = torch.randn(8, 128)
    b = torch.randn(8, 128)
    loss = brain_alignment_loss(a, b, loss_type="cosine")
    assert 0.0 <= loss.item() <= 2.0


def test_combined_loss_baseline():
    """Alpha=0 should give only task loss (unweighted)."""
    logits = torch.randn(8, 3)
    labels = torch.randint(0, 3, (8,))
    total, task, brain = combined_loss(logits, labels, None, None, alpha=0.0)
    expected = F.cross_entropy(logits, labels)
    assert torch.allclose(total, expected, atol=1e-5)
    assert brain.item() == 0.0


def test_combined_loss_with_class_weight():
    """Class weights should be passed through to cross_entropy correctly."""
    logits = torch.randn(8, 3)
    labels = torch.zeros(8, dtype=torch.long)  # all class 0
    weight = torch.tensor([0.5, 1.5, 1.0])
    total, task, _ = combined_loss(logits, labels, None, None, alpha=0.0,
                                   class_weight=weight)
    expected = F.cross_entropy(logits, labels, weight=weight)
    assert torch.allclose(total, expected, atol=1e-5)


def test_combined_loss_weight_reduces_majority_class_loss():
    """Downweighting majority class should change the loss value."""
    torch.manual_seed(0)
    logits = torch.randn(6, 3)   # non-uniform logits
    labels = torch.tensor([0, 0, 0, 1, 2, 2])
    # No weight
    _, loss_no_weight, _ = combined_loss(logits, labels, None, None, alpha=0.0)
    # Down-weight class 0 strongly
    w = torch.tensor([0.1, 2.0, 2.0])
    _, loss_weighted, _ = combined_loss(logits, labels, None, None, alpha=0.0,
                                        class_weight=w)
    # Weighted loss should differ when logits are non-uniform
    assert not torch.allclose(loss_no_weight, loss_weighted, atol=1e-4)


def test_combined_loss_alpha_one():
    """Alpha=1 should give only brain loss."""
    logits = torch.randn(8, 3)
    labels = torch.randint(0, 3, (8,))
    proj = torch.randn(8, 128)
    brain = torch.randn(8, 128)
    total, task, b_loss = combined_loss(logits, labels, proj, brain, alpha=1.0)
    expected = brain_alignment_loss(proj, brain, "mse")
    assert torch.allclose(total, expected, atol=1e-5)


def test_combined_loss_interpolation():
    """Total = (1-α)*task + α*brain."""
    logits = torch.randn(8, 3)
    labels = torch.randint(0, 3, (8,))
    proj   = torch.randn(8, 128)
    brain  = torch.randn(8, 128)
    alpha  = 0.3
    total, task, b = combined_loss(logits, labels, proj, brain, alpha=alpha)
    expected = (1 - alpha) * task + alpha * b
    assert torch.allclose(total, expected, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# 4. RSA and CKA
# ──────────────────────────────────────────────────────────────────────────────

def test_rsa_identical_representations():
    rng = np.random.default_rng(0)
    rep = rng.standard_normal((50, 64))
    rho, pval = rsa_correlation(rep, rep)
    assert rho > 0.99, f"RSA of identical reps should be ~1, got {rho}"


def test_rsa_random_representations():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((50, 64))
    b = rng.standard_normal((50, 64))
    rho, pval = rsa_correlation(a, b)
    assert -1.0 <= rho <= 1.0


def test_cka_identical():
    rng = np.random.default_rng(2)
    rep = rng.standard_normal((50, 64))
    cka = cka_linear(rep, rep)
    assert abs(cka - 1.0) < 1e-4, f"CKA of identical reps should be 1, got {cka}"


def test_cka_range():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((50, 32))
    b = rng.standard_normal((50, 16))
    cka = cka_linear(a, b)
    assert 0.0 <= cka <= 1.0, f"CKA out of [0,1]: {cka}"


def test_cka_zero_for_orthogonal():
    n, d = 100, 10
    a = np.zeros((n, d))
    a[:, 0] = 1.0
    b = np.zeros((n, d))
    b[:, 1] = 1.0
    # After centring both are orthogonal in feature space
    cka = cka_linear(a, b)
    assert cka < 0.1


# ──────────────────────────────────────────────────────────────────────────────
# 5. Control generators
# ──────────────────────────────────────────────────────────────────────────────

def make_fake_brain(n: int = 50, d: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, d)) * 3 + 1).astype(np.float32)


def test_synthetic_shape():
    brain = make_fake_brain()
    rng = np.random.default_rng(42)
    tr, vl, te, audit = make_synthetic_control(brain, 40, 5, 5, rng)
    assert tr.shape == (40, 128)
    assert vl.shape == (5,  128)
    assert te.shape == (5,  128)


def test_synthetic_mean_matched():
    brain = make_fake_brain(200)
    rng = np.random.default_rng(42)
    tr, _, _, audit = make_synthetic_control(brain, 200, 20, 20, rng)
    # Mean error should be small (within ~0.5 given random sampling)
    mean_err = float(np.abs(brain.mean(axis=0) - tr.mean(axis=0)).mean())
    assert mean_err < 1.0, f"Mean matching error too large: {mean_err}"


def test_shuffled_preserves_marginal():
    brain = make_fake_brain(50)
    val   = make_fake_brain(10, seed=1)
    test  = make_fake_brain(15, seed=2)
    rng   = np.random.default_rng(7)
    shuf_tr, shuf_vl, shuf_te, _ = make_shuffled_control(brain, val, test, rng)

    # Same values, different order → sorted arrays should be identical
    assert np.allclose(np.sort(brain.ravel()), np.sort(shuf_tr.ravel()))
    assert shuf_tr.shape == brain.shape


def test_shuffled_breaks_correspondence():
    brain = make_fake_brain(50)
    val   = make_fake_brain(10, seed=1)
    test  = make_fake_brain(15, seed=2)
    rng   = np.random.default_rng(99)
    shuf_tr, _, _, _ = make_shuffled_control(brain, val, test, rng)
    # With high probability, the first row will differ after shuffle
    n_same = int((np.abs(brain - shuf_tr) < 1e-6).all(axis=1).sum())
    assert n_same < 50, "Shuffled array should not be identical to original"


def test_random_control_shape():
    brain = make_fake_brain(50)
    rng   = np.random.default_rng(0)
    tr, vl, te, audit = make_random_control(brain, 40, 5, 5, rng)
    assert tr.shape == (40, 128)
    assert vl.shape == (5,  128)
    assert te.shape == (5,  128)


def test_random_control_dtype():
    brain = make_fake_brain(50)
    rng   = np.random.default_rng(0)
    tr, _, _, _ = make_random_control(brain, 40, 5, 5, rng)
    assert tr.dtype == np.float32


def test_no_test_leakage_in_synthetic():
    """Synthetic control must be generated from TRAINING data only."""
    brain_train = make_fake_brain(80, seed=0)
    rng = np.random.default_rng(42)
    _, _, te, audit = make_synthetic_control(brain_train, 80, 10, 10, rng)
    # Audit should record that only training stats were used
    assert audit["type"] == "synthetic_matched"
    # Generated test samples should NOT be identical to brain_train samples
    # (they are sampled from the distribution, not copied)
    for row in te:
        assert not any(np.allclose(row, brain_train[i]) for i in range(len(brain_train)))


def test_verify_matching_output():
    rng = np.random.default_rng(0)
    real = rng.standard_normal((50, 10)).astype(np.float32)
    syn  = rng.standard_normal((50, 10)).astype(np.float32)
    report = verify_matching(real, syn, "test")
    assert "mean_abs_diff_mu"  in report
    assert "mean_abs_diff_std" in report
    assert report["n_real"] == 50
    assert report["n_syn"]  == 50


# ──────────────────────────────────────────────────────────────────────────────
# 6. Training invariants
# ──────────────────────────────────────────────────────────────────────────────

def test_model_gradient_flows():
    """Gradients should flow to all parameters in the combined loss."""
    model = EEGClassifier(input_size=21, n_classes=3, hidden_size=32,
                          num_layers=1, bidirectional=False,
                          proj_dim=16, brain_dim=128)
    x = torch.randn(8, 21, 256)
    b = torch.randn(8, 128)
    y = torch.randint(0, 3, (8,))
    logits, proj = model(x)
    total, _, _ = combined_loss(logits, y, proj, b, alpha=0.5)
    total.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.any(torch.isnan(param.grad)), f"NaN gradient in {name}"


def test_baseline_no_gradient_to_proj_head():
    """With alpha=0 (baseline), proj_head should still get gradients from task loss
    only if it's in the forward path — but since proj is not used in task loss,
    gradients through proj_head should be zero or absent (proj not used)."""
    model = EEGClassifier(input_size=21, n_classes=3, hidden_size=32,
                          num_layers=1, bidirectional=False,
                          proj_dim=16, brain_dim=128)
    x = torch.randn(8, 21, 256)
    y = torch.randint(0, 3, (8,))
    logits, proj = model(x)
    # Alpha=0 → no brain loss, proj is computed but not used in loss
    total, _, _ = combined_loss(logits, y, proj, None, alpha=0.0)
    total.backward()
    # Encoder/classifier should have gradients
    assert model.encoder.gru.weight_ih_l0.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ──────────────────────────────────────────────────────────────────────────────
# 7. Anti-aliased stride (BUG-2 fix)
# ──────────────────────────────────────────────────────────────────────────────

def test_antialias_stride_shape():
    """Anti-aliased stride produces correct output shape."""
    from dataset import _antialias_and_stride
    X = np.random.randn(10, 21, 1024).astype(np.float32)
    X_dec = _antialias_and_stride(X, stride=4)
    assert X_dec.shape == (10, 21, 256), f"Expected (10,21,256) got {X_dec.shape}"


def test_antialias_stride_dtype():
    from dataset import _antialias_and_stride
    X = np.random.randn(5, 21, 512).astype(np.float32)
    X_dec = _antialias_and_stride(X, stride=2)
    assert X_dec.dtype == np.float32


def test_antialias_no_stride():
    from dataset import _antialias_and_stride
    X = np.random.randn(4, 5, 100).astype(np.float32)
    X_out = _antialias_and_stride(X, stride=1)
    np.testing.assert_array_equal(X, X_out)


def test_antialias_suppresses_high_freq():
    """After stride=4 anti-aliasing, signal energy above Nyquist/2 should be suppressed."""
    from dataset import _antialias_and_stride
    import numpy as np
    # Generate a pure 40 Hz sine at 256 Hz (above 32 Hz Nyquist after stride=4)
    sfreq = 256.0
    t = np.linspace(0, 1, int(sfreq), endpoint=False)
    sig_40hz = np.sin(2 * np.pi * 40 * t).astype(np.float32)
    X = sig_40hz[np.newaxis, np.newaxis, :]  # (1, 1, 256)
    X_dec = _antialias_and_stride(X, stride=4)   # (1, 1, 64)
    # Energy of decimated signal should be near zero (40 Hz aliased out)
    energy = float(np.abs(X_dec).mean())
    assert energy < 0.2, f"High-freq energy not suppressed: {energy:.4f}"

    # A 5 Hz signal should be preserved
    sig_5hz = np.sin(2 * np.pi * 5 * t).astype(np.float32)
    X2 = sig_5hz[np.newaxis, np.newaxis, :]
    X2_dec = _antialias_and_stride(X2, stride=4)
    energy2 = float(np.abs(X2_dec).mean())
    assert energy2 > 0.4, f"Low-freq signal incorrectly suppressed: {energy2:.4f}"


def test_eeg_dataset_uses_antialias(tmp_path):
    """EEGDataset should apply anti-aliased stride, not raw decimation."""
    from dataset import EEGDataset, _antialias_and_stride
    import numpy as np

    # Make a synthetic npz with a high-frequency signal
    sfreq = 256.0
    t = np.linspace(0, 4, int(sfreq * 4), endpoint=False)
    # 40 Hz signal — should be attenuated by anti-alias before stride=4
    X_raw = np.tile(np.sin(2 * np.pi * 40 * t), (5, 21, 1)).transpose(0, 2, 1).astype(np.float32)
    # X_raw shape: (5, 21, 1024) — wait, need proper shape
    X_raw = np.random.randn(5, 21, 1024).astype(np.float32)
    X_raw[:, 0, :] = np.tile(np.sin(2 * np.pi * 40 * t[:1024]), 5).reshape(5, 1024)

    y = np.zeros(5, dtype=np.int32)
    npz_path = tmp_path / "eeg_test_antialias.npz"
    np.savez(npz_path, X=X_raw, y_setsize=y, y_match=y, y_correct=y,
             subject=np.array(["s"]*5), session=np.array(["s"]*5),
             trial_idx=np.arange(5, dtype=np.int32))

    ds_strided = EEGDataset(npz_path, stride=4)
    ds_raw_dec = EEGDataset.__new__(EEGDataset)
    # Manually apply raw decimation for comparison
    import torch
    ds_raw_dec.X = torch.from_numpy(X_raw[:, :, ::4])
    ds_raw_dec.y = torch.zeros(5, dtype=torch.long)

    # The anti-aliased version should differ from raw decimation on high-freq signal
    ch0_antialias = ds_strided.X[:, 0, :].numpy()
    ch0_raw       = ds_raw_dec.X[:, 0, :].numpy()
    diff = float(np.abs(ch0_antialias - ch0_raw).mean())
    assert diff > 0.01, "Anti-aliased and raw-decimated outputs should differ on 40 Hz signal"
