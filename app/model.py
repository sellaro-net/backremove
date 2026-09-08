import logging
import os
from enum import Enum
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from withoutbg import WithoutBG
from withoutbg.models import OpenWeightsModel

logger = logging.getLogger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
VALID_DEVICES = {"auto", "cpu", "cuda"}
BIREFNET_REPOSITORY = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
BIREFNET_INPUT_SIZE = 1024


class ModelName(str, Enum):
    FAST = "fast"
    QUALITY = "quality"


class ModelUnavailableError(RuntimeError):
    pass


_model = None
_quality_model = None
_quality_preprocess = None
_inference_provider = "unloaded"


def _prepare_model_for_cuda(model_path: Path) -> Path:
    """Defuse optimized operators that CUDAExecutionProvider cannot execute."""
    import onnx
    from onnx import helper

    output_path = model_path.with_name(f"{model_path.stem}.cuda{model_path.suffix}")
    if (
        output_path.is_file()
        and output_path.stat().st_mtime >= model_path.stat().st_mtime
    ):
        return output_path

    model = onnx.load(str(model_path))
    new_nodes = []
    defused = 0

    for node in model.graph.node:
        if node.op_type != "FusedConv":
            new_nodes.append(node)
            continue

        activation = next(
            (
                attr.s.decode() if isinstance(attr.s, bytes) else attr.s
                for attr in node.attribute
                if attr.name == "activation"
            ),
            None,
        )
        if activation != "Sigmoid":
            new_nodes.append(node)
            continue

        conv_attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in node.attribute
            if attr.name != "activation"
        }
        intermediate = f"{node.output[0]}_pre_sigmoid"
        new_nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    node.input,
                    [intermediate],
                    f"{node.name}_conv",
                    **conv_attrs,
                ),
                helper.make_node(
                    "Sigmoid",
                    [intermediate],
                    node.output,
                    f"{node.name}_sigmoid",
                ),
            ]
        )
        defused += 1

    if defused == 0:
        return model_path

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    logger.info("Prepared CUDA-compatible model at %s", output_path)
    return output_path


class _CudaOpenWeightsModel(OpenWeightsModel):
    def _load_model(self) -> None:
        if self.model_path is None:
            raise RuntimeError("Model path was not resolved.")

        import torch

        # Import PyTorch first: its Windows loader also registers cuDNN sub-DLLs.
        ort.preload_dlls(directory=str(Path(torch.__file__).parent / "lib"))

        available = ort.get_available_providers()
        if CUDA_PROVIDER not in available:
            raise RuntimeError(
                f"{CUDA_PROVIDER} is unavailable. Available providers: {available}"
            )

        cuda_model_path = _prepare_model_for_cuda(self.model_path)
        self.session = ort.InferenceSession(
            str(cuda_model_path),
            providers=[
                (
                    CUDA_PROVIDER,
                    {"arena_extend_strategy": "kSameAsRequested"},
                )
            ],
        )
        self.session.disable_fallback()
        canvas_size = int(self.sidecar.get("canvas_size", 448))
        input_name = self.sidecar.get("input_name", "rgb")
        self.session.run(
            None,
            {
                input_name: np.zeros(
                    (1, 3, canvas_size, canvas_size),
                    dtype=np.float32,
                )
            },
        )
        active_provider = self.session.get_providers()[0]
        if active_provider != CUDA_PROVIDER:
            raise RuntimeError(f"Expected {CUDA_PROVIDER}, got {active_provider}.")


def _load_cpu_model():
    model = WithoutBG.open_weights()
    model.preload()
    return model


def load_model():
    global _model, _inference_provider
    if _model is not None:
        return _model

    requested_device = os.environ.get("INFERENCE_DEVICE", "auto").lower()
    if requested_device not in VALID_DEVICES:
        raise ValueError(
            f"Invalid INFERENCE_DEVICE={requested_device!r}; "
            f"expected one of {sorted(VALID_DEVICES)}."
        )

    logger.info("Loading fast model (requested device: %s)...", requested_device)
    use_cuda = requested_device == "cuda" or (
        requested_device == "auto" and CUDA_PROVIDER in ort.get_available_providers()
    )

    if use_cuda:
        try:
            model = _CudaOpenWeightsModel()
            model.preload()
            _model = model
            _inference_provider = CUDA_PROVIDER
        except Exception:
            if requested_device == "cuda":
                raise
            logger.warning(
                "CUDA initialization failed; falling back to CPU.",
                exc_info=True,
            )
            _model = _load_cpu_model()
            _inference_provider = CPU_PROVIDER
    else:
        _model = _load_cpu_model()
        _inference_provider = CPU_PROVIDER

    logger.info("Fast model ready (provider: %s).", _inference_provider)
    return _model


def _get_fast_model():
    if _model is None:
        raise RuntimeError("Fast model not loaded.")
    return _model


def _quality_enabled() -> bool:
    return os.environ.get("QUALITY_MODEL_ENABLED", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_quality_model():
    global _quality_model, _quality_preprocess
    if _quality_model is not None:
        return _quality_model
    if not _quality_enabled():
        raise ModelUnavailableError("The quality model is disabled on this deployment.")
    if os.environ.get("INFERENCE_DEVICE", "auto").lower() != "cuda":
        raise ModelUnavailableError("The quality model requires INFERENCE_DEVICE=cuda.")

    try:
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
    except ImportError as exc:
        raise ModelUnavailableError(
            "The quality model dependencies are not installed."
        ) from exc

    if not torch.cuda.is_available():
        raise ModelUnavailableError("PyTorch CUDA is unavailable.")

    logger.info(
        "Loading quality model %s at revision %s...",
        BIREFNET_REPOSITORY,
        BIREFNET_REVISION,
    )
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            BIREFNET_REPOSITORY,
            revision=BIREFNET_REVISION,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.float16,
        )
        model = model.to("cuda").eval()
        preprocess = transforms.Compose(
            [
                transforms.Resize((BIREFNET_INPUT_SIZE, BIREFNET_INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        with torch.inference_mode():
            model(
                torch.zeros(
                    (1, 3, BIREFNET_INPUT_SIZE, BIREFNET_INPUT_SIZE),
                    device="cuda",
                    dtype=torch.float16,
                )
            )[-1].sigmoid()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as exc:
        model = None
        torch.cuda.empty_cache()
        raise ModelUnavailableError(
            "Insufficient GPU memory to load the quality model."
        ) from exc
    except OSError as exc:
        model = None
        raise ModelUnavailableError(
            "The pinned quality model is not cached. Run setup-gpu.ps1."
        ) from exc

    _quality_model = model
    _quality_preprocess = preprocess
    logger.info("Quality model ready (PyTorch CUDA, FP16).")
    return _quality_model


def load_enabled_models() -> None:
    load_model()
    if _quality_enabled():
        load_quality_model()


def _remove_with_quality_model(input_image: Image.Image) -> Image.Image:
    model = load_quality_model()
    if _quality_preprocess is None:
        raise RuntimeError("Quality model preprocessing is unavailable.")

    import torch

    try:
        tensor = (
            _quality_preprocess(input_image)
            .unsqueeze(0)
            .to(device="cuda", dtype=torch.float16)
        )
        with torch.inference_mode():
            prediction = model(tensor)[-1].sigmoid()[0, 0]
            mask_array = prediction.float().cpu().numpy()
        del tensor, prediction
    except torch.cuda.OutOfMemoryError as exc:
        # Drop CUDA references before releasing unused cached blocks.
        tensor = None  # noqa: F841
        prediction = None  # noqa: F841
        torch.cuda.empty_cache()
        raise ModelUnavailableError(
            "Insufficient GPU memory to run the quality model."
        ) from exc

    mask = Image.fromarray(np.clip(mask_array * 255.0, 0, 255).astype(np.uint8))
    try:
        resized_mask = mask.resize(input_image.size, Image.Resampling.BILINEAR)
    finally:
        mask.close()
    result = input_image.copy()
    try:
        result.putalpha(resized_mask)
    except Exception:
        result.close()
        raise
    finally:
        resized_mask.close()
    return result


def remove_background(
    input_image: Image.Image,
    model_name: ModelName,
) -> Image.Image:
    if model_name is ModelName.FAST:
        return _get_fast_model().remove_background(input_image)
    if model_name is ModelName.QUALITY:
        return _remove_with_quality_model(input_image)
    raise ValueError(f"Unsupported model: {model_name!r}")


def get_inference_provider() -> str:
    return _inference_provider


def get_model_status() -> dict[str, dict[str, object]]:
    return {
        ModelName.FAST.value: {
            "loaded": _model is not None,
            "provider": _inference_provider,
            "runtime": "onnxruntime",
        },
        ModelName.QUALITY.value: {
            "enabled": _quality_enabled(),
            "loaded": _quality_model is not None,
            "provider": CUDA_PROVIDER if _quality_model is not None else None,
            "runtime": "pytorch",
            "precision": "fp16",
            "repository": BIREFNET_REPOSITORY,
            "revision": BIREFNET_REVISION,
        },
    }
