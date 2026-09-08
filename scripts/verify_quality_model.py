"""Exercise the pinned model with the supported PyTorch API on CPU, not CUDA."""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main() -> None:
    import huggingface_hub
    import torch
    import transformers
    from transformers import AutoModelForImageSegmentation

    from app.model import BIREFNET_REPOSITORY, BIREFNET_REVISION
    from app.prefetch_models import prefetch_quality_model

    # Match the native CUDA installation so CI cannot conceal model/API drift.
    if torch.__version__.split("+", 1)[0] != "2.14.0":
        raise RuntimeError("The compatibility check requires the supported PyTorch 2.14.0 API.")
    torch.set_num_threads(2)
    prefetch_quality_model()
    model = AutoModelForImageSegmentation.from_pretrained(
        BIREFNET_REPOSITORY,
        revision=BIREFNET_REVISION,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float32,
    ).to("cpu").eval()
    with torch.inference_mode():
        mask = model(torch.zeros((1, 3, 64, 64), dtype=torch.float32))[-1].sigmoid()
    if tuple(mask.shape) != (1, 1, 64, 64) or not torch.isfinite(mask).all().item():
        raise RuntimeError("The pinned model did not produce a finite 64x64 segmentation mask.")
    print(json.dumps({
        "repository": BIREFNET_REPOSITORY,
        "revision": BIREFNET_REVISION,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "torch": torch.__version__,
        "device": str(mask.device),
        "mask_shape": list(mask.shape),
        "finite": True,
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        seen: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            message = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted]", str(current))
            print(f"{type(current).__name__}: {message}", file=sys.stderr)
            current = current.__cause__ or current.__context__
        raise SystemExit(1)
