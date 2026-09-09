"""Offline-only BiRefNet exporter. Reuses the audited DeformConv and IEEE-FP16 helpers.

Sources and original helper hashes are recorded in sources.lock.json. The rejected
onnxconverter-common path from the exploratory script is intentionally not retained.
No GPU execution, remote model code download, or production Python dependency.
"""
import argparse
from collections import Counter
import gc
import hashlib
import heapq
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import warnings

PIN = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
sys.dont_write_bytecode = True


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def isolate_caches(out_dir, cpu_only=False):
    # Set before importing transformers/torch. No global cache gets modified,
    # even when transformers copies remote-code files or creates __init__.py.
    os.environ.update({
        "HF_HOME": str(out_dir / "hf-cache"),
        "HF_HUB_CACHE": str(out_dir / "hf-cache" / "hub"),
        "HF_MODULES_CACHE": str(out_dir / "hf-cache" / "modules"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "TORCH_HOME": str(out_dir / "torch-cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(out_dir / "torch-cache" / "inductor"),
    })
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_quality(snapshot):
    from transformers import AutoModelForImageSegmentation
    # A direct local snapshot plus an isolated HF_MODULES_CACHE prevents any
    # write into either the canonical hub snapshot or transformers module cache.
    return AutoModelForImageSegmentation.from_pretrained(
        str(Path(snapshot).resolve()), revision=PIN, trust_remote_code=True,
        local_files_only=True,
    ).eval()


def graph_nodes(graph):
    import onnx
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from graph_nodes(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for child in attr.graphs:
                    yield from graph_nodes(child)


def graph_record(model):
    import onnx
    def value_info(value):
        tensor = value.type.tensor_type
        return {
            "name": value.name, "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
            "shape": [dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                      for dim in tensor.shape.dim],
        }
    counts = Counter(f"{node.domain or 'ai.onnx'}::{node.op_type}"
                     for node in graph_nodes(model.graph))
    return {
        "ir_version": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "inputs": [value_info(item) for item in model.graph.input],
        "outputs": [value_info(item) for item in model.graph.output],
        "node_count": sum(counts.values()), "node_types": dict(sorted(counts.items())),
        "initializer_dtypes": dict(Counter(
            onnx.TensorProto.DataType.Name(item.data_type) for item in model.graph.initializer)),
        "nonstandard_nodes": [
            {"name": node.name, "domain": node.domain, "op_type": node.op_type}
            for node in graph_nodes(model.graph) if node.domain not in ("", "ai.onnx")
        ],
    }


def check_quality_io(model, dtype):
    import onnx
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Quality graph must have exactly one input and one output")
    for value, expected in ((model.graph.input[0], [1, 3, 1024, 1024]),
                            (model.graph.output[0], [1, 1, 1024, 1024])):
        tensor = value.type.tensor_type
        if tensor.elem_type != dtype or [dim.dim_value for dim in tensor.shape.dim] != expected:
            raise ValueError(f"Unexpected quality I/O: {value}")
    deform = [node for node in graph_nodes(model.graph) if node.op_type == "DeformConv"]
    if not deform or any(node.domain not in ("", "ai.onnx") for node in deform):
        raise ValueError("Expected standard-domain DeformConv nodes")
    onnx.checker.check_model(model)


def register_deform_symbolic():
    import torch
    from torch.onnx.symbolic_helper import parse_args

    # This is the torch.ops schema, NOT torchvision.ops.deform_conv2d's Python
    # argument order. Torchvision supplies a zero bias if bias=None.
    @parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "b")
    def symbolic(g, x, weight, offset, mask, bias, stride_h, stride_w,
                 pad_h, pad_w, dilation_h, dilation_w, weight_groups,
                 offset_groups, use_mask):
        inputs = [x, weight, offset, bias]
        if use_mask:
            inputs.append(mask)
        return g.op(
            "DeformConv", *inputs, group_i=weight_groups,
            offset_group_i=offset_groups, strides_i=[stride_h, stride_w],
            pads_i=[pad_h, pad_w, pad_h, pad_w],
            dilations_i=[dilation_h, dilation_w],
        )

    torch.onnx.register_custom_op_symbolic("torchvision::deform_conv2d", symbolic, 19)


def iter_float_tensors(graph, prefix="graph"):
    import onnx
    for tensor in graph.initializer:
        yield f"{prefix}/initializer/{tensor.name}", tensor
    for index, node in enumerate(graph.node):
        key = f"{prefix}/node/{node.name or str(index)}"
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                yield f"{key}/{attr.name}", attr.t
            elif attr.type == onnx.AttributeProto.TENSORS:
                for i, tensor in enumerate(attr.tensors):
                    yield f"{key}/{attr.name}/{i}", tensor
            elif attr.type == onnx.AttributeProto.GRAPH:
                yield from iter_float_tensors(attr.g, f"{key}/{attr.name}")
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for i, child in enumerate(attr.graphs):
                    yield from iter_float_tensors(child, f"{key}/{attr.name}/{i}")


def rounding_audit(model):
    import numpy as np
    import onnx
    result = {}
    for key, tensor in iter_float_tensors(model.graph):
        if tensor.data_type != onnx.TensorProto.FLOAT:
            continue
        values = onnx.numpy_helper.to_array(tensor)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            half = values.astype("<f2")
        finite = np.isfinite(values)
        abs_values = np.abs(values)
        result[key] = {
            "elements": values.size,
            "fp32_sha256": hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest(),
            "round_to_fp16_sha256": hashlib.sha256(half.tobytes()).hexdigest(),
            "original_nonfinite": int(np.count_nonzero(~finite)),
            "finite_to_infinity": int(np.count_nonzero(finite & np.isinf(half))),
            "nonzero_to_zero": int(np.count_nonzero(finite & (values != 0) & (half == 0))),
            "would_be_clipped_by_converter_defaults": int(np.count_nonzero(
                finite & (((abs_values > 0) & (abs_values < 1e-7)) | (abs_values > 1e4)))),
        }
        del values, half, finite, abs_values
    return result


def confirm_rounding(model, audit):
    import onnx
    observed = set()
    for key, tensor in iter_float_tensors(model.graph):
        if key not in audit:
            continue
        values = onnx.numpy_helper.to_array(tensor)
        if tensor.data_type == onnx.TensorProto.FLOAT16:
            actual = hashlib.sha256(values.astype("<f2", copy=False).tobytes()).hexdigest()
            expected = audit[key]["round_to_fp16_sha256"]
            audit[key]["converted_dtype"] = "FLOAT16"
        elif tensor.data_type == onnx.TensorProto.FLOAT:
            actual = hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()
            expected = audit[key]["fp32_sha256"]
            audit[key]["converted_dtype"] = "FLOAT"
        else:
            raise ValueError(f"Unexpected converted tensor dtype: {key}")
        if actual != expected:
            raise ValueError(f"Conversion changed tensor beyond IEEE fp16 rounding: {key}")
        observed.add(key)
        del values
    missing = set(audit) - observed
    if missing:
        raise ValueError(f"Original conversion tensors disappeared: {sorted(missing)[:10]}")
    return {
        "tensors_checked": len(observed),
        "converted_fp16_tensors": sum(row["converted_dtype"] == "FLOAT16" for row in audit.values()),
        "retained_fp32_tensors": sum(row["converted_dtype"] == "FLOAT" for row in audit.values()),
        "finite_to_infinity": sum(row["finite_to_infinity"] for row in audit.values()
                                  if row["converted_dtype"] == "FLOAT16"),
        "nonzero_to_zero": sum(row["nonzero_to_zero"] for row in audit.values()
                              if row["converted_dtype"] == "FLOAT16"),
        "no_nonrounding_weight_changes": True,
    }


def topological_sort(graph):
    import onnx
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                topological_sort(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    topological_sort(child)
    nodes = list(graph.node)
    producers = {name: index for index, node in enumerate(nodes) for name in node.output if name}
    dependants = [[] for _ in nodes]
    indegree = [0] * len(nodes)
    for index, node in enumerate(nodes):
        required = {producers[name] for name in node.input if name and name in producers}
        indegree[index] = len(required)
        for producer in required:
            dependants[producer].append(index)
    ready = [index for index, count in enumerate(indegree) if count == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        index = heapq.heappop(ready)
        ordered.append(nodes[index])
        for dependant in dependants[index]:
            indegree[dependant] -= 1
            if indegree[dependant] == 0:
                heapq.heappush(ready, dependant)
    if len(ordered) != len(nodes):
        raise ValueError('Cycle in converted graph')
    del graph.node[:]
    graph.node.extend(ordered)


def export_fp32(snapshot, output, lock, target):
    versions = lock["export_versions_linux" if target == "linux-cuda" else "export_versions"]
    for name, expected in versions.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Export-Dependency {name}: erwartet {expected}, gefunden {actual}")
    for name in ("config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors"):
        if sha256(snapshot / name) != lock["sources"]["quality-" + name]["sha256"]:
            raise ValueError(f"BiRefNet-Quelldatei verändert: {name}")
    if json.loads((snapshot / "config.json").read_text())["bb_pretrained"] is not False:
        raise ValueError("Unzulässiger Backbone-Download")
    import torch
    import torchvision
    import onnx
    register_deform_symbolic()

    class FinalMask(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, tensor):
            return self.model(tensor)[-1].sigmoid().reshape(1, 1, 1024, 1024)

    model = load_quality(snapshot).to(device="cpu", dtype=torch.float32)
    wrapper = FinalMask(model).eval()
    example = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32, device="cpu")
    with warnings.catch_warnings(record=True) as captured, torch.inference_mode():
        warnings.simplefilter("always")
        torch.onnx.export(
            wrapper, example, str(output), opset_version=19, dynamo=False,
            input_names=["input"], output_names=["mask"],
            dynamic_axes=None, do_constant_folding=True,
        )
    report = {"stage": "fp32-export", "opset": 19, "dynamo": False,
              "device": "cpu", "versions": versions,
              "warnings": list(dict.fromkeys(str(item.message) for item in captured))}
    del wrapper, model, example, captured
    gc.collect()
    graph = onnx.load(str(output))
    check_quality_io(graph, onnx.TensorProto.FLOAT)
    report.update(graph_record(graph))
    if report["nonstandard_nodes"] or report["node_types"].get("ai.onnx::DeformConv") != 20:
        raise ValueError("Quality-Export enthält nicht den erwarteten Standard-DeformConv-Graphen")
    return report


def fold_fp32(source, output, ort_python_root):
    # Separate process, with only the SHA-verified official CPU wheel at the
    # front of sys.path. Never import the machine's ORT/CUDA installation.
    sys.path.insert(0, str(ort_python_root))
    import onnxruntime as ort
    if ort.__version__ != "1.26.0" or not Path(ort.__file__).resolve().is_relative_to(ort_python_root):
        raise ValueError("Offline-Faltung benötigt die private ORT-1.26.0-CPU-Quelle")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(output)
    ort.InferenceSession(str(source), sess_options=options, providers=["CPUExecutionProvider"])
    return {"stage": "cpu-basic-fold", "runtime": ort.__version__,
            "optimization": "ORT_ENABLE_BASIC", "inference_executed": False}


def convert_fp16(source, output, converter_path, lock, target):
    import numpy as np
    import onnx
    converter_hash = lock["ort_converter_linux_sha256" if target == "linux-cuda" else "ort_converter_sha256"]
    if sha256(converter_path) != converter_hash:
        raise ValueError("ORT-FP16-Konverter hat einen unerwarteten Hash")
    spec = importlib.util.spec_from_file_location("pinned_ort_float16", converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    graph = onnx.load(str(source))
    audit = rounding_audit(graph)
    if any(row["original_nonfinite"] or row["finite_to_infinity"] for row in audit.values()):
        raise ValueError("Quality-Gewichte sind nicht sicher als FP16 darstellbar")
    expected_half = {}
    for key, tensor in iter_float_tensors(graph.graph):
        if tensor.data_type == onnx.TensorProto.FLOAT:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                expected_half[key] = onnx.numpy_helper.to_array(tensor).astype("<f2").tobytes()
    converted = converter.convert_float_to_float16(
        graph, keep_io_types=False, disable_shape_infer=False,
    )
    del graph
    gc.collect()
    restored = 0
    for key, tensor in iter_float_tensors(converted.graph):
        if key in expected_half and tensor.data_type == onnx.TensorProto.FLOAT16:
            actual = onnx.numpy_helper.to_array(tensor).astype("<f2", copy=False).tobytes()
            if actual != expected_half[key]:
                for field in ("float_data", "int32_data", "int64_data", "double_data",
                              "uint64_data", "string_data", "external_data"):
                    tensor.ClearField(field)
                tensor.data_location = onnx.TensorProto.DEFAULT
                tensor.raw_data = expected_half[key]
                restored += 1
    del expected_half
    gc.collect()
    rounding = confirm_rounding(converted, audit)
    dump_json(output.with_suffix(".rounding.json"), audit)
    topological_sort(converted.graph)
    check_quality_io(converted, onnx.TensorProto.FLOAT16)
    onnx.checker.check_model(converted, full_check=True)
    report = graph_record(converted)
    if report["nonstandard_nodes"] or report["node_types"].get("ai.onnx::DeformConv") != 20:
        raise ValueError("Quality-FP16 enthält nicht ausschließlich Standardoperatoren")
    if report["node_count"] != 2692 or rounding["converted_fp16_tensors"] != 626:
        raise ValueError("Quality-FP16 weicht strukturell vom geprüften gefalteten Modell ab")
    onnx.save(converted, str(output))
    report.update({"stage": "ort-fp16", "rounding": rounding,
                   "full_type_check": True, "tensors_restored_to_ieee_rounding": restored,
                   "converter": file_record(converter_path),
                   "rounding_policy": "IEEE binary16; no arbitrary tensor clamping",
                   "reference_sha256": lock["quality_reference_sha256"],
                   "byte_identical_to_reference": sha256(output) == lock["quality_reference_sha256"]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["export", "fold", "half"])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ort-python-root", type=Path)
    parser.add_argument("--converter", type=Path)
    parser.add_argument("--target", required=True, choices=["windows-cuda", "linux-cuda"])
    args = parser.parse_args()
    platform = "linux" if args.target == "linux-cuda" else "win32"
    if sys.platform != platform or sys.version_info[:2] != (3, 12):
        parser.error("Quality-Export benötigt natives Python 3.12 auf der Zielplattform")
    output = args.output.resolve()
    source = args.source.resolve()
    if output.exists() or output == source:
        raise ValueError("Ausgabe muss ein neues privates Artefakt sein")
    output.parent.mkdir(parents=True, exist_ok=True)
    isolate_caches(output.parent / "offline-caches", cpu_only=True)
    lock = json.loads(Path(__file__).with_name("sources.lock.json").read_text())
    if args.stage == "export":
        report = export_fp32(source, output, lock, args.target)
    elif args.stage == "fold":
        if args.ort_python_root is None:
            parser.error("--ort-python-root fehlt")
        report = fold_fp32(source, output, args.ort_python_root.resolve())
    else:
        if args.converter is None:
            parser.error("--converter fehlt")
        report = convert_fp16(source, output, args.converter.resolve(), lock, args.target)
    report.update({"status": "ok", "artifact": file_record(output),
                   "script": file_record(__file__), "python": sys.version, "target": args.target})
    dump_json(output.with_suffix(".provenance.json"), report)
    print(json.dumps({"status": "ok", "output": str(output), "sha256": sha256(output)}), flush=True)


if __name__ == "__main__":
    main()
