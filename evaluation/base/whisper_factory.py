from evaluation.onnxruntime.onnx_encoder import ONNXWhisperEncoder
from evaluation.onnxruntime.onnx_decoder import ONNXWhisperDecoder
from evaluation.hailo.hailo_sdk_encoder import HailoSdkWhisperEncoder
from evaluation.hailo.hailo_sdk_decoder import HailoSdkWhisperDecoder
import os


def get_backend_from_file_extension(model_path):
    root, ext = os.path.splitext(model_path)
    ext = ext[1:]
    if ext == "onnx":
        backend = "onnx"
    elif ext == "har":
        backend = "hailo"
    else:
        raise ValueError(f"Unsupported model extension: {backend}")
        
    return backend


def get_encoder(model_path, target="native"):
    """Returns the appropriate encoder based on the backend."""
    backend = get_backend_from_file_extension(model_path)
    if backend == "onnx":
        if target != "native":
            print(f"Selected target {target} for encoder will be ignored when using ONNXRuntime")
        return ONNXWhisperEncoder(model_path)
    elif backend == "hailo":
        print(f"Encoder target: {target}")
        return HailoSdkWhisperEncoder(model_path, target)
    else:
        raise ValueError(f"Unsupported encoder backend: {backend}")


def get_decoder(model_path, variant="tiny", target="native"):
    """Returns the appropriate decoder based on the backend."""
    backend = get_backend_from_file_extension(model_path)
    if backend == "onnx":
        if target != "native":
            print(f"Selected target {target} for decoder will be ignored when using ONNXRuntime")
        return ONNXWhisperDecoder(model_path, variant)
    elif backend == "hailo":
        print(f"Decoder target: {target}")
        return HailoSdkWhisperDecoder(model_path, variant, target)
    else:
        raise ValueError(f"Unsupported decoder backend: {backend}")
