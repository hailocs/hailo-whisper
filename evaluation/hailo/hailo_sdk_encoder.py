from evaluation.base.whisper_runner import WhisperEncoder
from hailo_sdk_client import ClientRunner, InferenceContext


class HailoSdkWhisperEncoder(WhisperEncoder):

    def __init__(self, encoder_model_path: str, target="native"):
        """
        :param encoder_model_path: Path to the encoder model file.
        """
        super().__init__(encoder_model_path)
        self.encoder_runner = ClientRunner(har=self.encoder_model_path)
        self.backend = "hailo"
        self.target = target
        self.input_audio_length = self._get_input_length_from_hn()

    def _get_input_length_from_hn(self):
        # Determine input length from model input shape
        layers = self.encoder_runner.get_hn()["layers"]
        for layer_name in layers:
            if "input_layer1" in layer_name:
                input_audio_length = int(layers[layer_name]["input_shapes"][0][2] / 100)
                break
        return input_audio_length

    def encode(self, input_mel):
        encoder_inputs = input_mel
        if self.target == "quantized":
            with self.encoder_runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
                encoder_outputs = self.encoder_runner.infer(ctx, encoder_inputs)
        elif self.target == "hw":
            with self.encoder_runner.infer_context(InferenceContext.SDK_HAILO_HW) as ctx:
                encoder_outputs = self.encoder_runner.infer(ctx, encoder_inputs)
        else:
            with self.encoder_runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
                encoder_outputs = self.encoder_runner.infer(ctx, encoder_inputs)
        return encoder_outputs