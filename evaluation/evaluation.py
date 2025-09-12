from evaluation.base.whisper_factory import get_encoder, get_decoder
from common.preprocessing import preprocess, improve_input_audio
import whisper
import argparse
import os
import sys
from evaluation.wer_calculator import calc_wer
from common.postprocessing import clean_transcription
from tqdm import tqdm
from common.log_utils import logger


def get_args():
    """
    Initialize and run the argument parser.

    Return:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Encoder/Decoder inference emualtion and evaluation")
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["tiny", "tiny.en", "base", "base.en"],
        help="Whisper model variant to run"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help="Whisper encoder path (ONNX / HAR)"
    )
    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        help="Whisper decoder path (ONNX / HAR)"
    )
    parser.add_argument(
        "--encoder-target",
        type=str,
        default="native",
        choices=["native", "quantized", "hw"],
        help="Optional target for encoder, when using Hailo SDK (default: native)"
    )
    parser.add_argument(
        "--decoder-target",
        type=str,
        default="native",
        choices=["native", "quantized", "hw"],
        help="Optional target for decoder, when using Hailo SDK (default: native)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to run evaluation on (default: 50)"
    )

    return parser.parse_args()


def parse_gt_file(gt_file_path):
    gt = {}
    with open(gt_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                file_id, transcription = parts
                gt[file_id] = transcription
    return gt


def create_evaluation_dict_librispeech(librispeech_audio_dir, chunk_length, target_size=512, is_nhwc=True):
    logger.info(f"Selected target size: {target_size}")
    max_duration = chunk_length  # use only files shorter than chunk_length 
    sample_rate = 16000
    list_dir = os.listdir(librispeech_audio_dir)

    def build_eval_dataset():
        eval_dataset = {}
        gt = {}
        for dir in list_dir:
            dir_path = os.path.join(librispeech_audio_dir, dir)
            list_subdir = os.listdir(dir_path)

            for subdir in list_subdir:
                subdir_path = os.path.join(dir_path, subdir)
                list_files = os.listdir(subdir_path)

                gt_file = [f for f in os.listdir(subdir_path) if f.endswith(".txt")]  # find ground-truth file
                if len(gt_file) == 0:
                    logger.error(f"No ground-truth file found in {subdir_path}")
                    sys.exit()
                gt_file_path = os.path.join(subdir_path, gt_file[0])
                subdir_gt = parse_gt_file(gt_file_path)

                for elem in list_files:
                    root, ext = os.path.splitext(elem)
                    if ext == ".flac":
                        file_path = os.path.join(subdir_path, elem)
                        audio = whisper.load_audio(file_path)
                        duration = audio.shape[0] / sample_rate

                        if duration > max_duration:
                            continue

                        logger.info(f"Preprocessing {file_path} with duration {duration:.2f} seconds")

                        single_audio_mels = preprocess(audio, is_nhwc=is_nhwc, chunk_length=chunk_length)
                        eval_dataset[file_path] = single_audio_mels

                        gt[file_path] = subdir_gt[root]

                    if len(eval_dataset) >= target_size:
                        return eval_dataset, gt  # exits all loops

        return eval_dataset, gt

    eval_dataset, gt = build_eval_dataset()
    logger.info(f"Created evaluation set with {len(eval_dataset)} entries")
    return eval_dataset, gt


def main():
    args = get_args()
    variant = args.variant

    test_set_dir = "audio/test-clean/LibriSpeech/test-clean"
    if not os.path.exists(test_set_dir):
        logger.error(f"Test set directory {test_set_dir} does not exist. Please run the following script to download the LibriSpeech test-clean dataset.\n")
        logger.error(">> cd ./audio")
        logger.error(">> ./download_test_set.sh")
        sys.exit(1)

    whisper_encoder = get_encoder(args.encoder, target=args.encoder_target)
    whisper_decoder = get_decoder(args.decoder, variant=variant, target=args.decoder_target)

    if whisper_encoder.backend == "onnx":
        is_nhwc = False
    else:
        is_nhwc = True

    chunk_length = whisper_encoder.get_input_length()

    eval_dataset, gt = create_evaluation_dict_librispeech(test_set_dir, chunk_length=chunk_length, target_size=args.num_samples, is_nhwc=is_nhwc)

    #audio, start_time = improve_input_audio(audio, vad=True, low_audio_gain=True)

    total_wer = 0.0
    for filepath, mel in tqdm(eval_dataset.items(), desc="Processing", total=len(eval_dataset)):
        encoded_features = whisper_encoder.encode(mel[0])
        # print(encoded_features.shape)
        transcription = whisper_decoder.decode(encoded_features)
        cleaned_transcription = clean_transcription(transcription[0])
        measures = calc_wer(cleaned_transcription, gt[filepath])
        total_wer += measures.wer
        # print(f"\nT: {cleaned_transcription}\nGT: {gt[filepath]}")
        # print(f"\nWord Error Rate: {100 * measures.wer:.3f} %")
    avg_wer = total_wer / len(eval_dataset)
    logger.info(f"\nAverage Word Error Rate on {len(eval_dataset)} samples: {100 * avg_wer:.3f} %")

    return


if __name__ == "__main__":
    main()
