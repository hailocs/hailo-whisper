#!/usr/bin/env python3

import onnxruntime as ort
import numpy as np
import os
import sys
from transformers import AutoTokenizer
from conversion.tokenization import extract_tokenization
from conversion.utils.conversion_utils import get_encoder_input_length_from_onnx, get_decoder_sequence_length_from_onnx
from tqdm import tqdm

import argparse
from common.log_utils import logger

DATASET_SIZE = 1024  # Size of the calibration set to create


def get_args():
    """
    Initialize and run the argument parser.

    Return:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Whisper decoder conversion script for Hailo")
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["tiny", "tiny.en", "base", "base.en"],
        help="Whisper model variant to create the dataset for (default: tiny)"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Whisper ONNX encoder path"
    )
    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Whisper ONNX decoder path"
    )
    return parser.parse_args()


class WhisperONNX:

    def __init__(self, variant, encoder_model_path: str, decoder_model_path: str):
        """
        Initialize the WhisperONNX model.

        :param encoder_model_path: Path to the encoder model file.
        :param decoder_model_path: Path to the decoder model file.
        """
        self.encoder_model_path = encoder_model_path
        self.decoder_model_path = decoder_model_path
        self.encoder_session = None
        self.decoder_session = None
        self.tokenization_session = None
        self.tokenizer = None
        self.variant = variant
        self.chunk_length = get_encoder_input_length_from_onnx(self.encoder_model_path)
        self.decoding_sequence_length = get_decoder_sequence_length_from_onnx(self.decoder_model_path)
        self.encoder_calib_set_path = f"./conversion/optimization/{variant}/encoder_calib_set_{variant}_{self.chunk_length}s.npy"
        self.decoder_calib_set_output_path = f"./conversion/optimization/{variant}/decoder_calib_set_{variant}_{self.chunk_length}s_seq_{self.decoding_sequence_length}.npz"
        self.token_embedding_weight = None
        self.onnx_add_input = None
        self.constant_output_0 = np.array([1])  # Unsqueeze axis
        self.decoder_model_name = os.path.splitext(os.path.basename(self.decoder_model_path))[0]

        self._load_model()
        self._load_tokenizer()

    def _load_model(self):
        """
        Load the ONNX model using ONNX Runtime.
        """
        self.encoder_session = ort.InferenceSession(self.encoder_model_path, providers=["CPUExecutionProvider"])
        self.decoder_session = ort.InferenceSession(self.decoder_model_path, providers=["CPUExecutionProvider"])

        # Load the tokenization assets, or create them if not available
        base_path = os.path.dirname(os.path.abspath(__file__))
        token_embedding_path = os.path.join(base_path, f"optimization/{self.variant}/token_embedding_weight_{self.variant}_seq_{self.decoding_sequence_length}.npy")
        onnx_add_input_path = os.path.join(base_path, f"optimization/{self.variant}/onnx_add_input_{self.variant}_seq_{self.decoding_sequence_length}.npy")
        if not os.path.exists(token_embedding_path) or not os.path.exists(onnx_add_input_path):
            logger.error(f"Tokenization assets not found for {self.variant}. Extracting them...")
            extract_tokenization(self.variant, self.decoder_model_path)
        self.token_embedding_weight = np.load(token_embedding_path)
        self.onnx_add_input = np.load(onnx_add_input_path)

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai/whisper-" + self.variant)
    
    def run_encoder(self, input_mel):
        encoder_input_name = self.encoder_session.get_inputs()[0].name
        encoder_inputs = {encoder_input_name: input_mel.astype(np.float32)}
        encoder_outputs = self.encoder_session.run(None, encoder_inputs)
        encoded_features = encoder_outputs[0]  # Extract the encoder output
        return encoded_features

    def run_decoder(self, encoded_features):

        # Assuming <|startoftranscript|> is the start-of-sequence token, which should be encoded as a scalar
        start_token_id = self.tokenizer.encode('<|startoftranscript|>', truncation=True, padding=False, add_special_tokens=False)
        decoder_input_ids = np.array([[start_token_id[0]]], dtype=np.int64)  # Shape (1,1)
        decoder_input_ids = np.concatenate([decoder_input_ids, np.zeros((1, self.decoding_sequence_length - 1), dtype=np.int64)], axis=1)

        generated_tokens = []

        # Run Decoder Iteratively
        for i in range(self.decoding_sequence_length - 1):
            decoder_inputs = {
                'decoder_input_ids': decoder_input_ids,       # Token IDs (int64)
                'encoder_hidden_states': encoded_features  # Encoder output (float32)
            }

            decoder_outputs = self.decoder_session.run(None, decoder_inputs)
            next_token = np.argmax(decoder_outputs[0][:, i])  # Get most probable token. Look just at the i index in the second dimension (current prediction)

            generated_tokens.append(next_token)

            # Update decoder input
            # Fix: Append the next token and reshape (keep batch size 1)

            decoder_input_ids[0][i + 1] = next_token  # Shape (1, seq_len+1)

            if next_token == self.tokenizer.eos_token_id:
                break

        full_token_sequence = decoder_input_ids  # check

        return full_token_sequence

    def run_tokenization_onnx(self, full_token_sequence):
        session_inputs = {
            'decoder_input_ids': full_token_sequence
        }

        tokenization_outputs = self.tokenization_session.run(None, session_inputs)
        tokenization_outputs_nhwc = np.transpose(tokenization_outputs[0], [0, 2, 3, 1])
        return tokenization_outputs_nhwc

    def run_tokenization_npy(self, full_token_sequence):
        decoder_input_ids = full_token_sequence
        # embedding lookup
        gather_output = self.token_embedding_weight[decoder_input_ids]  # Shape: (len(decoder_input_ids), 384)
        # Add bias
        add_output = gather_output + self.onnx_add_input  # Broadcasting with shape (32, 384)
        # insert dimension at axis=1
        unsqueeze_output = np.expand_dims(add_output, axis=int(self.constant_output_0[0]))  # Shape: (32, 1, 384)
        # Transpose (0, 3, 2, 1) + turn into NHWC (0, 2, 3, 1)
        tokenization_outputs_nhwc = np.transpose(unsqueeze_output, (0, 2, 1, 3))
        return tokenization_outputs_nhwc

    def run(self, audio_path: str):
        # Load the audio
        if not os.path.exists(self.encoder_calib_set_path):
            logger.error(f"Encoder calibration set not found at {self.encoder_calib_set_path}.")
            logger.error("It is generated during the encoder conversion process.")
            sys.exit()
        loaded_dataset = np.load(self.encoder_calib_set_path)
        mel_spectrograms = np.transpose(loaded_dataset, [0, 3, 1, 2])  # the encoder calib set data are in NHWC, but we are using ONNXRT in this case
        mel_spectrograms = mel_spectrograms[:DATASET_SIZE]  # Limit the data to get from the encoder calib set

        features = []
        tokenized_ids = []
        for mel in tqdm(mel_spectrograms, desc="Processing", total=len(mel_spectrograms)):
            mel = np.expand_dims(mel, axis=0)
            encoded_features = self.run_encoder(mel)
            features.append(encoded_features)
            full_token_sequence = self.run_decoder(encoded_features)
            tokenization_outputs = self.run_tokenization_npy(full_token_sequence)
            tokenized_ids.append(tokenization_outputs)

        features = np.stack(features, axis=0)  # since the elements are 3D arrays
        tokenized_ids = np.concatenate(tokenized_ids, axis=0)  # since the elemnts are already 4D arrays
        calib_set_dict = {
            self.decoder_model_name.replace(".", "_") + '/input_layer1': features,
            self.decoder_model_name.replace(".", "_") + '/input_layer2': tokenized_ids
        }
        np.savez(self.decoder_calib_set_output_path, **calib_set_dict)
        logger.info(f"Decoder calibration set saved to {self.decoder_calib_set_output_path}")

        return


def main():
    args = get_args()
    if not os.path.exists(args.encoder):
        logger.error(f"Encoder model path {args.encoder} does not exist")
        sys.exit(1)
    if not os.path.exists(args.decoder):
        logger.error(f"Decoder model path {args.decoder} does not exist")
        sys.exit(1)

    logger.info(f"Creating decoder calibration set for variant: {args.variant}")

    whisper_pipeline = WhisperONNX(variant=args.variant, encoder_model_path=args.encoder, decoder_model_path=args.decoder)
    whisper_pipeline.run("NULL")

    return


if __name__ == "__main__":
    main()
