import argparse
import torch
import onnx
import os
import re
from onnxsim import simplify
from common.log_utils import logger


def get_args():
    parser = argparse.ArgumentParser(description="Script to export Whisper model to ONNX format")
    parser.add_argument(
        "--variant",
        type=str,
        default="tiny",
        choices=["tiny", "tiny.en", "base", "base.en"],
        help="Variant of the Whisper odel to export (default: tiny)"
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=None,
        help="Model input audio length in seconds"
    )
    parser.add_argument(
        "--decoder-sequence-length",
        type=int,
        default=None,
        help="Decoder token sequence length"
    )
    return parser.parse_args()


def set_scaling_factor(scaling_factor):
    # Path to PyTorch source code
    model_path = 'third_party/whisper/whisper/model.py'

    pattern = re.compile(r'(self\.dims\.n_audio_ctx\s*//\s*)\d+')

    with open(model_path, 'r') as f:
        lines = f.readlines()

    # Replace in lines
    new_lines = [
        pattern.sub(lambda m: f"{m.group(1)}{scaling_factor}", line)
        for line in lines
    ]

    with open(model_path, 'w') as f:
        f.writelines(new_lines)


def export_model(variant, input_length, decoder_sequence_length, encoder_input):

    original_frames_length = 3000  # original model works on 30 seconds-long input audio
    original_encoder_seq_len = 1500

    if (variant == "tiny" or variant == "tiny.en"):
        default_input_length = 10
        default_decoder_sequence_length = 32  # Maximum number of tokens
        hidden_states_channels = 384
    elif (variant == "base" or variant == "base.en"):
        default_input_length = 5
        default_decoder_sequence_length = 24
        hidden_states_channels = 512
    else:  # placeholder for other variants
        default_input_length = 5
        default_decoder_sequence_length = 24
        hidden_states_channels = 512

    if input_length is None:
        input_length = default_input_length

    if decoder_sequence_length is None:
        decoder_sequence_length = default_decoder_sequence_length

    scaling_factor = int(original_frames_length / (input_length * 100))
    encoder_seq_len = int(original_encoder_seq_len / scaling_factor)
    suffix = f"{input_length}s"
    mel_frames_length = int(original_frames_length / scaling_factor)

    logger.info(f"Exporting model {variant} with the following parameters")
    logger.info(f"Input audio length: {input_length} seconds")
    logger.info(f"Decoder sequence length: {decoder_sequence_length} tokens")

    logger.info(f"Setting audio_ctx scaling factor to {scaling_factor}")
    set_scaling_factor(scaling_factor)
    import whisper  # importing whisper only after setting the scaling_factor

    model = whisper.load_model(variant)
    device = "cpu"
    model.to(device)
    model.eval()
    encoder_name = variant + "-whisper-encoder-" + suffix
    encoder_onnx_path = os.path.join("./export", encoder_name + ".onnx")
    torch.onnx.export(model.encoder, torch.randn(1, encoder_input, 1, mel_frames_length).to("cpu"), encoder_onnx_path)
    model_onnx = onnx.load(encoder_onnx_path)
    model_simp, check = simplify(model_onnx)
    onnx.save(model_simp, encoder_onnx_path)

    logger.info(f"Encoder exported to {encoder_onnx_path}")

    # Generate dummy inputs for export
    batch_size = 1
    hidden_size = model.dims.n_text_state  # Getting the value based on the Whisper variant

    encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, hidden_size, dtype=torch.float32)
    decoder_input_ids = torch.cat([torch.tensor([[50258]], dtype=torch.int64), torch.zeros((1, decoder_sequence_length - 1), dtype=torch.int64)], dim=1)

    # Move to the same device as model
    encoder_hidden_states = encoder_hidden_states.to(device)
    decoder_input_ids = decoder_input_ids.to(device)

    decoder = model.decoder

    decoder_name = f"{variant}-whisper-decoder-{input_length}s-seq-{decoder_sequence_length}.onnx"
    decoder_onnx_path = os.path.join("./export", decoder_name)

    torch.onnx.export(
        decoder,
        (decoder_input_ids, encoder_hidden_states),  # Input tuple
        decoder_onnx_path,
        opset_version=13,
        input_names=["decoder_input_ids", "encoder_hidden_states"],
        output_names=["logits"],
    )

    logger.info(f"Decoder exported to {decoder_onnx_path}")

    input_shapes = {
        "decoder_input_ids": [1, decoder_sequence_length],
        "encoder_hidden_states": [1, encoder_seq_len, hidden_states_channels]
    }
    model_onnx = onnx.load(decoder_onnx_path)
    model_simp, check = simplify(model_onnx, overwrite_input_shapes=input_shapes)
    onnx.save(model_simp, decoder_onnx_path)

    # Check if the simplification was successful
    if check:
        logger.info("ONNX model was successfully simplified!")
    else:
        logger.info("ONNX model simplification failed!")


if __name__ == "__main__":
    # Export the encoder and decoder models
    args = get_args()
    export_model(args.variant, args.input_length, args.decoder_sequence_length, encoder_input=80)

