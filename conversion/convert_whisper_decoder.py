#!/usr/bin/env python3

import os
import numpy as np
from hailo_sdk_client import ClientRunner
import sys
import argparse
from conversion.utils.conversion_utils import get_decoder_sequence_length_from_onnx, get_input_audio_length_from_decoder_onnx
from common.log_utils import logger


def get_conversion_args():
    """
    Initialize and run the argument parser.

    Return:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Whisper decoder conversion script for Hailo")
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
        help="Whisper model variant to convert"
    )
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo8",
        choices=["hailo8", "hailo8l", "hailo10h"],
        help="Hardware architecture to use (default: hailo8)"
    )
    return parser.parse_args()


##################

def get_model_details(decoder_onnx_path, variant, hw_arch):
    if not ("tiny" in variant) and not ("base" in variant):
        logger.error(f"Unknown variant: {variant}")
        sys.exit(1)
    decoder_sequence_length = get_decoder_sequence_length_from_onnx(decoder_onnx_path)
    input_audio_length = get_input_audio_length_from_decoder_onnx(decoder_onnx_path)
    output_dir = f"./conversion/converted/{variant}_whisper_decoder_{input_audio_length}s_seq_{decoder_sequence_length}_{hw_arch}"
    return output_dir, input_audio_length, decoder_sequence_length


class HailoWhisperDecoder:

    def __init__(self, decoder_onnx_path, variant="tiny", hw_arch="hailo8"):
        self.variant = variant
        self.hw_arch = hw_arch

        self.decoder_onnx_path = decoder_onnx_path
        self.output_dir, input_audio_length, decoder_sequence_length = get_model_details(self.decoder_onnx_path, variant, hw_arch)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.model_script = f"./conversion/optimization/{variant}/{hw_arch}/decoder_model_script_{variant}_matmul_split.alls"
        self.calib_set_path = f"./conversion/optimization/{variant}/decoder_calib_set_{variant}_{input_audio_length}s_seq_{decoder_sequence_length}.npz"

        self.model_name = os.path.splitext(os.path.basename(self.decoder_onnx_path))[0]

        try:
            self.runner = ClientRunner(hw_arch=hw_arch)
        except Exception as e:
            logger.error(f"Error initializing ClientRunner with architecture '{hw_arch}': {e}.")
            sys.exit*()

        self.har_path = os.path.join(self.output_dir, self.model_name + ".har")
        self.har_path_optimized = os.path.join(self.output_dir, self.model_name + "_optimized.har")
        self.har_path_compiled = os.path.join(self.output_dir, self.model_name + "_compiled.har")
        self.hef_path = os.path.join(self.output_dir, self.model_name + ".hef")

    def parse_models(self):
        logger.title("\nModel parsing\n")
        self.runner.translate_onnx_model(self.decoder_onnx_path, self.model_name,
                                         start_node_names=["encoder_hidden_states", "/Reshape"],
                                         end_node_names=["/MatMul", "/MatMul_1", "/MatMul_2", "/MatMul_3"])
        self.runner.save_har(self.har_path)
        return

    def load_calibration_set(self):
        if not os.path.exists(self.calib_set_path):
            logger.error(f"Calibration set not found at {self.calib_set_path}.")
            logger.error("Please run the create_decoder_calib_set.py script to generate it.")
            sys.exit()
        logger.info(f"Loading calibration set from {self.calib_set_path}")
        loaded_dataset = np.load(self.calib_set_path, allow_pickle=True)
        calib_dataset_dict = {}
        for key in loaded_dataset.files:
            calib_dataset_dict[key] = loaded_dataset[key]
        return calib_dataset_dict

    def quantize_model(self):
        logger.title("\nModel Optimization\n")
        calib_data = self.load_calibration_set()
        self.runner.load_model_script(self.model_script)
        self.runner.optimize(
            calib_data=calib_data,
            work_dir=self.output_dir)
        self.runner.save_har(self.har_path_optimized)
        return

    def compile_model(self):
        logger.title("\nModel Compilation\n")
        hef = self.runner.compile()
        with open(self.hef_path, 'wb') as f:
            f.write(hef)
        self.runner.save_har(self.har_path_compiled)
        return

    def run(self):
        self.parse_models()
        self.quantize_model()
        self.compile_model()
        return


def main():
    args = get_conversion_args()
    if not os.path.exists(args.model_path):
        logger.error(f"Model path {args.model_path} does not exist")
        sys.exit(1)
    print("Model conversion started")
    converter = HailoWhisperDecoder(decoder_onnx_path=args.model_path, variant=args.variant, hw_arch=args.hw_arch)
    converter.run()
    print("Model conversion completed")
    return


if __name__ == '__main__':
    exit(main())
