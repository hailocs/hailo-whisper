import onnxruntime as ort


def get_encoder_input_length_from_onnx(encoder_model_path: str):
    encoder_session = ort.InferenceSession(encoder_model_path, providers=["CPUExecutionProvider"])
    input_audio_length = int(encoder_session.get_inputs()[0].shape[3] / 100)
    return input_audio_length


def get_input_audio_length_from_decoder_onnx(decoder_model_path: str):
    decoder_session = ort.InferenceSession(decoder_model_path, providers=["CPUExecutionProvider"])
    input_audio_length = int(decoder_session.get_inputs()[1].shape[1] / 50)
    return input_audio_length


def get_decoder_sequence_length_from_onnx(decoder_model_path: str):
    decoder_session = ort.InferenceSession(decoder_model_path, providers=["CPUExecutionProvider"])
    sequence_length = int(decoder_session.get_outputs()[0].shape[1])
    return sequence_length