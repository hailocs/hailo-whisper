import onnxruntime as ort
import numpy as np
from evaluation.base.whisper_runner import WhisperEncoder


class ONNXWhisperEncoder(WhisperEncoder):

    def __init__(self, encoder_model_path: str):
        """
        :param encoder_model_path: Path to the encoder model file.
        """
        super().__init__(encoder_model_path)
        self.backend = "onnx"
        self.encoder_session = ort.InferenceSession(self.encoder_model_path, providers=["CPUExecutionProvider"])
        self.input_audio_length = int(self.encoder_session.get_inputs()[0].shape[3] / 100)

    def encode(self, input_mel):
        encoder_input_name = self.encoder_session.get_inputs()[0].name
        encoder_inputs = {encoder_input_name: input_mel.astype(np.float32)}
        encoder_outputs = self.encoder_session.run(None, encoder_inputs)
        encoded_features = encoder_outputs[0]  # Extract the encoder output

        return encoded_features
