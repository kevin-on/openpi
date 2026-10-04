"""Checkpoint dtype checks using tiny parameters; no model weights or tokenizers."""

from types import SimpleNamespace

from flax import nnx
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import policy_config


class _TinyModel(nnx.Module):
    def __init__(self):
        self.frozen = nnx.Param(jnp.zeros(2, dtype=jnp.float32))
        self.trainable = nnx.Param(jnp.zeros(2, dtype=jnp.float32))


class _TinyModelConfig:
    load = _model.BaseModelConfig.load

    def create(self, rng):
        del rng
        return _TinyModel()


@pytest.mark.parametrize("preserve_dtype", [False, True], ids=["default_bfloat16", "stored_dtypes"])
def test_jax_checkpoint_parameter_dtype(tmp_path, monkeypatch, preserve_dtype):
    # These float32 values lose information if restored through bfloat16 first.
    original = {
        "frozen": np.asarray([1.5, -2.0], dtype=jnp.bfloat16),
        "trainable": np.asarray([1.001, -0.1234567], dtype=np.float32),
    }
    rounded = original["trainable"].astype(jnp.bfloat16)
    assert not np.array_equal(original["trainable"], rounded.astype(np.float32))
    with ocp.PyTreeCheckpointer(use_ocdbt=False) as checkpointer:
        checkpointer.save(tmp_path / "params", {"params": original})

    data = SimpleNamespace(
        data_transforms=transforms.Group(),
        model_transforms=transforms.Group(),
        use_quantile_norm=False,
    )
    config = SimpleNamespace(
        model=_TinyModelConfig(),
        data=SimpleNamespace(create=lambda *_: data),
        assets_dirs=(),
        policy_metadata={},
    )
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda _: tmp_path)
    monkeypatch.setattr(policy_config.checkpoint_config, "with_assets", lambda cfg, _: cfg)
    # The actual Orbax restore and BaseModelConfig.load run; no inference is needed.
    monkeypatch.setattr(policy_config._policy, "Policy", lambda model, **_: SimpleNamespace(model=model))
    kwargs = {"params_dtype": None} if preserve_dtype else {}
    policy = policy_config.create_trained_policy(config, tmp_path, norm_stats={}, **kwargs)
    loaded = nnx.state(policy.model).to_pure_dict()

    assert loaded["frozen"].dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(loaded["frozen"]), original["frozen"])
    assert loaded["trainable"].dtype == (jnp.float32 if preserve_dtype else jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(loaded["trainable"]), original["trainable"] if preserve_dtype else rounded)
