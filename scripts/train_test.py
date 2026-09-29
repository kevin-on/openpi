import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    args = [config_name, "--batch-size=2", "--checkpoint-base-dir="+str(tmp_path/"checkpoint"),
            "--exp-name=test", "--num-train-steps=2", "--log-interval=1", "--no-overwrite", "--no-resume"]
    train.main(_config.cli(args), config_args=args)
    args = [a for a in args if a not in ("--no-resume", "--num-train-steps=2")]+["--resume", "--num-train-steps=4"]
    train.main(_config.cli(args), config_args=args)
