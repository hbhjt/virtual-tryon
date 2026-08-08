from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = ROOT / "vendor" / "CatVTON"
MODEL_CACHE = ROOT / "models" / "catvton-cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CatVTON on CPU in an isolated environment.")
    parser.add_argument("--person", required=True)
    parser.add_argument("--garment", required=True)
    parser.add_argument("--target-mask", required=True)
    parser.add_argument("--hands-mask", required=True)
    parser.add_argument("--sleeve-mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _cached_file(pattern: str) -> Path | None:
    files = list(
        (MODEL_CACHE / "hub" / "models--zhengchong--CatVTON" / "snapshots").glob(pattern)
    )
    return files[0] if files else None


def main() -> None:
    args = parse_args()
    if not VENDOR_DIR.exists():
        raise SystemExit(f"CatVTON source is missing: {VENDOR_DIR}")
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(VENDOR_DIR))
    sys.path.insert(0, str(ROOT))

    import cv2
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image

    from app.ai_masks import (
        build_generation_mask,
        composite_to_original,
        harmonize_garment_color,
        include_color_matched_old_edges,
        letterbox_image,
        letterbox_mask,
        parsed_old_clothes_mask,
        refine_mask_with_generated_parse,
    )
    from model.SCHP import SCHP

    worker_started = time.perf_counter()
    torch.set_num_threads(max(1, min(16, (os.cpu_count() or 8) - 2)))
    attention_file = _cached_file("*/mix-48k-1024/attention/model.safetensors")
    lip_checkpoint = _cached_file("*/SCHP/exp-schp-201908261155-lip.pth")
    atr_checkpoint = _cached_file("*/SCHP/exp-schp-201908301523-atr.pth")
    if not all((attention_file, lip_checkpoint, atr_checkpoint)):
        snapshot_download(
            repo_id="zhengchong/CatVTON",
            allow_patterns=[
                "mix-48k-1024/attention/model.safetensors",
                "SCHP/exp-schp-201908261155-lip.pth",
                "SCHP/exp-schp-201908301523-atr.pth",
            ],
        )
        attention_file = _cached_file("*/mix-48k-1024/attention/model.safetensors")
        lip_checkpoint = _cached_file("*/SCHP/exp-schp-201908261155-lip.pth")
        atr_checkpoint = _cached_file("*/SCHP/exp-schp-201908301523-atr.pth")
    if not all((attention_file, lip_checkpoint, atr_checkpoint)):
        raise RuntimeError("CatVTON or SCHP checkpoint is incomplete")

    original = np.array(Image.open(args.person).convert("RGB"), dtype=np.uint8)
    target_mask = np.array(Image.open(args.target_mask).convert("L"), dtype=np.uint8)
    hands_mask = np.array(Image.open(args.hands_mask).convert("L"), dtype=np.uint8)
    sleeve_mask = np.array(Image.open(args.sleeve_mask).convert("L"), dtype=np.uint8)

    # Parse one model at a time and release it before CatVTON is constructed.
    parsing_started = time.perf_counter()
    lip_model = SCHP(str(lip_checkpoint), device="cpu")
    with torch.inference_mode():
        lip_parse = np.array(lip_model(Image.fromarray(original)), dtype=np.uint8)
    del lip_model
    gc.collect()

    atr_model = SCHP(str(atr_checkpoint), device="cpu")
    with torch.inference_mode():
        atr_parse = np.array(atr_model(Image.fromarray(original)), dtype=np.uint8)
    del atr_model
    gc.collect()
    parsing_seconds = time.perf_counter() - parsing_started

    original_clothes = parsed_old_clothes_mask(lip_parse, atr_parse)
    generation_mask = build_generation_mask(target_mask, lip_parse, atr_parse, hands_mask)
    del lip_parse, atr_parse
    gc.collect()

    person_fit, transform = letterbox_image(original, args.width, args.height)
    mask_fit = letterbox_mask(generation_mask, transform)
    original_clothes_fit = letterbox_mask(original_clothes, transform)
    sleeve_mask_fit = letterbox_mask(sleeve_mask, transform)
    garment_original = np.array(Image.open(args.garment).convert("RGB"), dtype=np.uint8)
    garment_fit, _ = letterbox_image(garment_original, args.width, args.height)

    from model.pipeline import CatVTONPipeline

    load_started = time.perf_counter()
    attention_checkpoint = str(attention_file.parents[2])
    pipeline = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt=attention_checkpoint,
        attn_ckpt_version="mix",
        weight_dtype=torch.float32,
        device="cpu",
        compile=False,
        skip_safety_check=True,
        use_tf32=False,
    )
    load_seconds = time.perf_counter() - load_started

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = pipeline(
            image=Image.fromarray(person_fit),
            condition_image=Image.fromarray(garment_fit),
            mask=Image.fromarray(mask_fit),
            num_inference_steps=args.steps,
            guidance_scale=2.5,
            height=args.height,
            width=args.width,
            generator=generator,
        )[0]
    generation_seconds = time.perf_counter() - generation_started

    post_parse_started = time.perf_counter()
    generated_rgb_raw = np.array(generated.convert("RGB"), dtype=np.uint8)
    del pipeline
    gc.collect()
    post_parse_model = SCHP(str(lip_checkpoint), device="cpu")
    with torch.inference_mode():
        generated_lip_parse = np.array(
            post_parse_model(Image.fromarray(generated_rgb_raw)),
            dtype=np.uint8,
        )
    del post_parse_model
    gc.collect()
    final_mask_fit = refine_mask_with_generated_parse(
        mask_fit,
        generated_lip_parse,
        original_clothes_fit,
        sleeve_mask_fit,
    )
    final_mask_fit = include_color_matched_old_edges(
        final_mask_fit,
        mask_fit,
        person_fit,
        original_clothes_fit,
    )
    post_parse_seconds = time.perf_counter() - post_parse_started
    postprocess_started = time.perf_counter()
    output = Path(args.output)
    if os.environ.get("CATVTON_SAVE_DEBUG") == "1":
        generated.save(output.with_name(f"{output.stem}-raw.png"), format="PNG")
        Image.fromarray(final_mask_fit).save(output.with_name(f"{output.stem}-mask.png"), format="PNG")
        Image.fromarray(garment_fit).save(output.with_name(f"{output.stem}-condition.png"), format="PNG")
    generated_rgb = harmonize_garment_color(
        generated_rgb_raw,
        final_mask_fit,
        garment_fit,
    )
    result = composite_to_original(
        original,
        generated_rgb,
        final_mask_fit,
        transform,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(output, format="PNG", compress_level=3)
    postprocess_seconds = time.perf_counter() - postprocess_started
    print(
        json.dumps(
            {
                "output": str(output),
                "parsing_seconds": round(parsing_seconds, 2),
                "post_parse_seconds": round(post_parse_seconds, 2),
                "load_seconds": round(load_seconds, 2),
                "generation_seconds": round(generation_seconds, 2),
                "postprocess_seconds": round(postprocess_seconds, 2),
                "worker_seconds": round(time.perf_counter() - worker_started, 2),
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
