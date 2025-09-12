import onnx
import numpy as np
import argparse
import os
import sys
from conversion.utils.conversion_utils import get_decoder_sequence_length_from_onnx


variant_to_add_input = {  # verify these in your model
    "tiny": "onnx::Add_1079",
    "tiny.en": "onnx::Add_1079",
    "base": "onnx::Add_1567",
    "base.en": "onnx::Add_1567"
}


def get_args():
    """
    Initialize and run the argument parser.

    Return:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Extract tokenization section from decoder model.")
    parser.add_argument(
        "model_path",
        type=str,
        help="Decoder ONNX model path"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="tiny",
        choices=["tiny", "tiny.en", "base", "base.en"],
        help="Whisper model variant to extract tokenization from (default: tiny)"
    )
    return parser.parse_args()


# Helper function to extract tensors from initializers
def get_initializer(name, model):
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(tuple(initializer.dims))
    raise ValueError(f"Initializer {name} not found in the model.")


def extract_tokenization(variant, model_path):

    model = onnx.load(model_path)
    # Extract the token embedding weight and bias
    token_embedding_weight = get_initializer("token_embedding.weight", model)  # Shape: (51865, 384)
    onnx_add_input = get_initializer(variant_to_add_input[variant], model)  # Shape: (seq_len, 384)

    # Print shapes to verify
    print("token_embedding_weight shape:", token_embedding_weight.shape)
    print("onnx_add_input shape:", onnx_add_input.shape)

    decoder_seq_len = get_decoder_sequence_length_from_onnx(model_path)
    token_embedding_output_path = f"./conversion/optimization/{variant}/token_embedding_weight_{variant}_seq_{decoder_seq_len}.npy"
    onnx_add_input_output_path = f"./conversion/optimization/{variant}/onnx_add_input_{variant}_seq_{decoder_seq_len}.npy"
    np.save(token_embedding_output_path, token_embedding_weight)
    np.save(onnx_add_input_output_path, onnx_add_input)
    print(f"Saved token_embedding_weight to {token_embedding_output_path}")
    print(f"Saved onnx_add_input to {onnx_add_input_output_path}")


def main():
    args = get_args()
    if not os.path.exists(args.model_path):
        print(f"Model path {args.model_path} does not exist")
        sys.exit(1)
    extract_tokenization(args.variant, args.model_path)
    return


if __name__ == "__main__":
    main()