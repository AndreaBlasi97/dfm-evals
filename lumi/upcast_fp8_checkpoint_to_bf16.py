#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from safetensors import safe_open
from safetensors.torch import save_file


QUANT_SCALE_SUFFIXES = (".qscale_weight", ".qscale_act")
QUANT_CONFIG_KEYS_TO_DROP = ("quantization_config",)
MISTRAL_PARAMS_KEYS_TO_DROP = ("quantization",)
FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)


@dataclass(frozen=True)
class ConversionStats:
    shard_name: str
    converted_tensors: int
    copied_tensors: int
    skipped_scale_tensors: int
    bytes_written: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upcast a Mistral FP8 safetensors checkpoint into BF16 shard-by-shard."
        )
    )
    parser.add_argument("--source", required=True, help="Source checkpoint directory")
    parser.add_argument("--output", required=True, help="Output checkpoint directory")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip shards that already exist in the output directory",
    )
    parser.add_argument(
        "--only-shards",
        nargs="*",
        default=None,
        help="Optional list of shard filenames to convert",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    return parser.parse_args()


def load_index(index_path: Path) -> dict:
    with index_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SystemExit(
            f"output directory is not empty: {output_dir} (pass --overwrite to reuse)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def patch_config(config_path: Path, output_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    for key in QUANT_CONFIG_KEYS_TO_DROP:
        config.pop(key, None)
    config["torch_dtype"] = "bfloat16"
    write_json(output_path, config)


def patch_mistral_params(params_path: Path, output_path: Path) -> None:
    with params_path.open("r", encoding="utf-8") as fh:
        params = json.load(fh)
    for key in MISTRAL_PARAMS_KEYS_TO_DROP:
        params.pop(key, None)
    write_json(output_path, params)


def copy_support_files(source_dir: Path, output_dir: Path) -> None:
    for entry in source_dir.iterdir():
        if entry.name.endswith(".safetensors"):
            continue
        if entry.name.endswith(".index.json"):
            continue
        if entry.name == "config.json":
            patch_config(entry, output_dir / entry.name)
            continue
        if entry.name == "params.json":
            patch_mistral_params(entry, output_dir / entry.name)
            continue
        target = output_dir / entry.name
        if entry.is_symlink():
            resolved = entry.resolve()
            if target.exists():
                target.unlink()
            os.symlink(resolved, target)
        elif entry.is_file():
            shutil.copy2(entry, target)


def select_shards(index_data: dict, only_shards: list[str] | None) -> list[str]:
    shards = sorted(set(index_data["weight_map"].values()))
    if not only_shards:
        return shards
    requested = set(only_shards)
    missing = sorted(requested - set(shards))
    if missing:
        raise SystemExit(f"unknown shards requested: {missing}")
    return [shard for shard in shards if shard in requested]


def dequantize_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return weight.to(torch.bfloat16) * scale.to(torch.bfloat16)


def convert_shard(source_path: Path, output_path: Path) -> ConversionStats:
    tensors: dict[str, torch.Tensor] = {}
    converted = 0
    copied = 0
    skipped_scales = 0

    with safe_open(source_path, framework="pt") as fh:
        keys = list(fh.keys())
        key_set = set(keys)
        metadata = fh.metadata()
        for key in keys:
            if key.endswith(QUANT_SCALE_SUFFIXES):
                skipped_scales += 1
                continue

            tensor = fh.get_tensor(key)
            if tensor.dtype in FP8_DTYPES:
                if not key.endswith(".weight"):
                    raise RuntimeError(f"unexpected FP8 tensor without weight suffix: {key}")
                scale_key = key[: -len(".weight")] + ".qscale_weight"
                if scale_key not in key_set:
                    raise RuntimeError(f"missing scale tensor for {key}: {scale_key}")
                scale = fh.get_tensor(scale_key)
                tensors[key] = dequantize_weight(tensor, scale)
                converted += 1
            else:
                tensors[key] = tensor
                copied += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=output_path.parent,
        prefix=output_path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as tmp_fh:
        tmp_path = Path(tmp_fh.name)
    try:
        save_file(tensors, str(tmp_path), metadata=metadata)
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return ConversionStats(
        shard_name=source_path.name,
        converted_tensors=converted,
        copied_tensors=copied,
        skipped_scale_tensors=skipped_scales,
        bytes_written=output_path.stat().st_size,
    )


def build_output_index(
    source_dir: Path,
    output_dir: Path,
    index_data: dict,
    shard_names: list[str],
) -> dict:
    new_weight_map: dict[str, str] = {}
    total_size = 0

    for shard_name in shard_names:
        source_path = source_dir / shard_name
        output_path = output_dir / shard_name
        if not output_path.exists():
            raise RuntimeError(f"missing converted shard: {output_path}")

        total_size += output_path.stat().st_size
        with safe_open(source_path, framework="pt") as fh:
            for key in fh.keys():
                if key.endswith(QUANT_SCALE_SUFFIXES):
                    continue
                new_weight_map[key] = shard_name

    return {
        "metadata": {"total_size": total_size},
        "weight_map": new_weight_map,
    }


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()

    if not source_dir.is_dir():
        raise SystemExit(f"source directory does not exist: {source_dir}")

    index_candidates = [
        source_dir / "consolidated.safetensors.index.json",
        source_dir / "model.safetensors.index.json",
    ]
    index_path = next((p for p in index_candidates if p.exists()), None)
    if index_path is None:
        raise SystemExit(f"no safetensors index found in {source_dir}")

    ensure_output_dir(output_dir, overwrite=args.overwrite)
    copy_support_files(source_dir, output_dir)

    index_data = load_index(index_path)
    shard_names = select_shards(index_data, args.only_shards)

    conversion_manifest: list[dict[str, int | str]] = []

    for shard_name in shard_names:
        source_path = source_dir / shard_name
        output_path = output_dir / shard_name
        if args.skip_existing and output_path.exists():
            print(f"SKIP {shard_name} existing={output_path}")
            continue

        print(f"CONVERT {source_path} -> {output_path}")
        stats = convert_shard(source_path, output_path)
        conversion_manifest.append(
            {
                "shard_name": stats.shard_name,
                "converted_tensors": stats.converted_tensors,
                "copied_tensors": stats.copied_tensors,
                "skipped_scale_tensors": stats.skipped_scale_tensors,
                "bytes_written": stats.bytes_written,
            }
        )
        print(
            "DONE"
            f" shard={stats.shard_name}"
            f" converted={stats.converted_tensors}"
            f" copied={stats.copied_tensors}"
            f" skipped_scales={stats.skipped_scale_tensors}"
            f" bytes={stats.bytes_written}"
        )

    output_index = build_output_index(source_dir, output_dir, index_data, shard_names)
    write_json(output_dir / index_path.name, output_index)

    write_json(
        output_dir / "upcast_manifest.json",
        {
            "source": str(source_dir),
            "output": str(output_dir),
            "source_index": index_path.name,
            "converted_shards": shard_names,
            "manifest": conversion_manifest,
        },
    )

    print(f"OUTPUT {output_dir}")


if __name__ == "__main__":
    main()
