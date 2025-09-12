from evaluation.base.whisper_factory import get_encoder, get_decoder
from common.preprocessing import preprocess, improve_input_audio
import whisper
import argparse
import os
from common.log_utils import logger


def get_args():
    """
    Initialize and run the argument parser.

    Return:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Encoder/Decoder inference emulation script")
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
        help="Optional target for encoder"
    )
    parser.add_argument(
        "--decoder-target",
        type=str,
        default="native",
        choices=["native", "quantized", "hw"],
        help="Optional target for decoder"
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Audio file path for the test (if None, uses a default file)"
    )

    return parser.parse_args()


def main():
    args = get_args()
    variant = args.variant

    whisper_encoder = get_encoder(args.encoder, target=args.encoder_target)

    chunk_length = whisper_encoder.get_input_length()

    whisper_decoder = get_decoder(args.decoder, variant=variant, target=args.decoder_target)

    if whisper_encoder.backend == "onnx":
        is_nhwc = False
    else:
        is_nhwc = True

    if args.audio_path is None:
        audio_path = "audio/dev-clean/LibriSpeech/dev-clean/84/121123/84-121123-0020.flac"
        logger.warning(f"No audio path provided, using default: {audio_path}")
    else:
        if not os.path.exists(args.audio_path):
            raise FileNotFoundError(f"Provided audio file not found: {args.audio_path}")
        audio_path = args.audio_path

    # Load the audio
    audio = whisper.load_audio(audio_path)

    audio, start_time = improve_input_audio(audio, vad=False, low_audio_gain=True)

    mel_spectrograms = preprocess(audio=audio, is_nhwc=is_nhwc, chunk_length=chunk_length, chunk_offset=start_time)
    for mel in mel_spectrograms:
        encoded_features = whisper_encoder.encode(mel)
        # print(encoded_features.shape)
        transcription = whisper_decoder.decode(encoded_features)
        print(transcription)
    return


if __name__ == "__main__":
    main()
