"""SFT recipes: replay CLI using the normal parser, then verify resolved settings."""
import dataclasses
import enum
import json
import os
from pathlib import Path
import re
import subprocess
from openpi.training import config as configs


def describe(value):
    """JSON comparison document, never a Python object deserializer."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, enum.Enum):
        return describe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(v) for v in value]
    if isinstance(value, re.Pattern):
        return {"pattern": value.pattern, "flags": value.flags}
    if isinstance(value, type):
        return {"type": value.__module__ + "." + value.__qualname__}
    name = type(value).__module__ + "." + type(value).__qualname__
    if dataclasses.is_dataclass(value):
        return {"type": name, "fields": {f.name: describe(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if callable(value) and hasattr(value, "__qualname__"):
        if "<lambda>" in value.__qualname__ or "<locals>" in value.__qualname__:
            raise TypeError("Config recipes cannot fingerprint anonymous functions")
        return {"callable": value.__module__ + "." + value.__qualname__}
    if hasattr(value, "__dict__"):
        return {"type": name, "fields": describe(vars(value))}
    raise TypeError(f"Unsupported config value: {name}")


def source_commit():
    if value := os.environ.get("OPENPI_SOURCE_COMMIT"):
        return value
    try:
        return subprocess.check_output(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unrecorded"


def make_record(config, args):
    args = list(args)
    replayed = configs.cli(args)
    resolved = describe(config)
    if describe(replayed) != resolved:
        raise ValueError("CLI does not reproduce the effective SFT config")
    return dict(version=2, config_args=args, resolved=resolved,
                openpi_commit=source_commit())


def restore(record):
    if record.get("version") != 2 or not isinstance(record.get("config_args"), list) or "resolved" not in record:
        raise ValueError("Checkpoint needs verified SFT config metadata (version 2); migrate explicitly")
    try:
        config = configs.cli(record["config_args"])
    except SystemExit as exc:
        raise ValueError("Saved SFT CLI is incompatible with this source") from exc
    if describe(config) != record["resolved"]:
        raise ValueError("SFT preset/config changed since checkpoint creation")
    return config


def read_record(checkpoint):
    return json.loads((Path(checkpoint) / "assets/config.json").read_text())


def load(checkpoint):
    return restore(read_record(checkpoint))


def with_assets(config, checkpoint):
    return dataclasses.replace(config, data=dataclasses.replace(config.data,
        assets=dataclasses.replace(config.data.assets, assets_dir=str(Path(checkpoint)/"assets"))))


def check_resume(config, checkpoint):
    """Allow operational changes, but require the saved model/data/training contract."""
    previous = load(checkpoint)
    for field in ('model', 'data', 'freeze_filter', 'optimizer', 'lr_schedule', 'ema_decay'):
        if describe(getattr(config, field)) != describe(getattr(previous, field)):
            raise ValueError(f'SFT resume config mismatch: {field}')
