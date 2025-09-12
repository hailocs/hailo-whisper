#!/usr/bin/env python3

import os
import numpy as np
from hailo_sdk_client import ClientRunner
import whisper
import sys
from common.preprocessing import preprocess
from conversion.utils.conversion_utils import get_encoder_input_length_from_onnx
import argparse
from common.log_utils import logger


def get_conversion_args():
    """
    Initialize and run the argument parser.

    Return:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Whisper encoder conversion script for Hailo")
    parser.add_argument(
        "model_path",
        type=str,
        help="Encoder ONNX model path"
    )
    parser.add_argument(
        "--load-calib-set",
        action="store_true",
        help="Reuse the previously generated calibration set (default: False)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
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


def get_model_details(encoder_onnx_path, variant, hw_arch):
    if "tiny" in variant:
        dataset_size = 4000
    elif "base" in variant:
        dataset_size = 2048
    else:
        logger.error(f"Unknown variant: {variant}")
        sys.exit(1)
    chunk_length = get_encoder_input_length_from_onnx(encoder_onnx_path)
    output_dir = f"./conversion/converted/{variant}_whisper_encoder_{chunk_length}s_{hw_arch}"
    return chunk_length, output_dir, dataset_size


librispeech_audio_dir = "./audio/dev-clean/LibriSpeech/dev-clean"


def create_calibration_set_librispeech(chunk_length, target_size=2048):
    logger.info("Creating encoder calibration set...")
    calib_dataset = []
    list_dir = os.listdir(librispeech_audio_dir)
    for dir in list_dir:
        dir_path = os.path.join(librispeech_audio_dir, dir)
        list_subdir = os.listdir(dir_path)
        for subdir in list_subdir:
            subdir_path = os.path.join(dir_path, subdir)
            list_files = os.listdir(subdir_path)
            for elem in list_files:
                root, ext = os.path.splitext(elem)
                if ext == ".flac":
                    file_path = os.path.join(subdir_path, elem)
                    print(file_path)
                    audio = whisper.load_audio(file_path)
                    single_audio_mels = preprocess(audio, is_nhwc=True, chunk_length=chunk_length)  # this returns a list
                    for mel in single_audio_mels:
                        calib_dataset.append(mel)
        if len(calib_dataset) > target_size:  # limit the dataset size
            break

    calib_dataset = np.concatenate(calib_dataset, axis=0)  # (N, H, W, C)
    logger.info(f"Created calibration set with shape {calib_dataset.shape}")
    return calib_dataset


class HailoWhisperEncoder:

    def __init__(self, encoder_onnx_path, variant="tiny", hw_arch="hailo8", load_calib_set=False):
        self.variant = variant
        self.hw_arch = hw_arch
        self.load_calib_set = load_calib_set

        self.encoder_onnx_path = encoder_onnx_path
        self.chunk_length, self.output_dir, self.dataset_size = get_model_details(self.encoder_onnx_path, variant, hw_arch)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.model_script = f"./conversion/optimization/{variant}/{hw_arch}/encoder_model_script_{variant}.alls"
        self.calib_set_path = f"./conversion/optimization/{variant}/encoder_calib_set_{variant}_{self.chunk_length}s.npy"

        self.model_name = os.path.splitext(os.path.basename(self.encoder_onnx_path))[0]
        
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
        self.runner.translate_onnx_model(self.encoder_onnx_path, self.model_name)
        self.runner.save_har(self.har_path)
        return

    def quantize_model(self):
        logger.title("\nModel Optimization\n")
        if self.load_calib_set:
            if os.path.exists(self.calib_set_path):
                logger.info(f"Loading calibration set from {self.calib_set_path}")
                calib_data = np.load(self.calib_set_path)
            else:
                logger.error(f"Calibration set not found at {self.calib_set_path}.")
        else:
            calib_data = create_calibration_set_librispeech(self.chunk_length, target_size=self.dataset_size)
            np.save(self.calib_set_path, calib_data)

        self.runner.load_model_script(self.model_script)
        self.runner.optimize(
            calib_data=calib_data,
            work_dir=self.output_dir)
        self.runner.save_har(self.har_path_optimized)
        return

    def compile_model(self):
        logger.title("\nModel compilation\n")
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
    logger.title("\nModel conversion started.")
    logger.info(f"Variant: {args.variant}")
    logger.info(f"Hardware Architecture: {args.hw_arch}")
    converter = HailoWhisperEncoder(encoder_onnx_path=args.model_path, variant=args.variant, hw_arch=args.hw_arch, load_calib_set=args.load_calib_set)
    converter.run()
    logger.title("Model conversion completed")
    return


if __name__ == '__main__':
    exit(main())
