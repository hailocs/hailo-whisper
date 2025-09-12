import onnxruntime as ort
import numpy as np
from evaluation.base.whisper_runner import WhisperDecoder
from transformers import AutoTokenizer
from common.postprocessing import apply_repetition_penalty


class ONNXWhisperDecoder(WhisperDecoder):

    def __init__(self, decoder_model_path: str, variant="tiny"):
        """
        :param decoder_model_path: Path to the encoder model file.
        """
        super().__init__(decoder_model_path)
        self.decoder_session = ort.InferenceSession(self.decoder_model_path, providers=["CPUExecutionProvider"])
        self.backend = "onnx"
        self.tokenizer = None
        self.variant = variant

        self.decoding_sequence_length = int(self.decoder_session.get_outputs()[0].shape[1])
        self._load_tokenizer()

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai/whisper-" + self.variant)

    def adjust_input_shape(self, decoder_input_data):
        if len(decoder_input_data.shape) == 4:   # in case data are encoded by Hailo 
            decoder_input_data = np.squeeze(decoder_input_data, axis=0)

        return decoder_input_data

    def decode(self, encoded_features):

        transcriptions = []
        encoded_features = self.adjust_input_shape(encoded_features)
        # Assuming <|startoftranscript|> is the start-of-sequence token, which should be encoded as a scalar
        # start_token_id = self.tokenizer.encode('<|startoftranscript|>', truncation=True, padding=False, add_special_tokens=False)
        start_token_id = [50258]

        decoder_input_ids = np.array([[start_token_id[0]]], dtype=np.int64)  # Shape (1,1)
        decoder_input_ids = np.concatenate([decoder_input_ids, np.zeros((1, self.decoding_sequence_length - 1), dtype=np.int64)], axis=1)

        generated_tokens = []

        # print("Decoder Input Names:", [inp.name for inp in self.decoder_session.get_inputs()])
        # print("Decoder Input Shapes:", [inp.shape for inp in self.decoder_session.get_inputs()])

        # Run Decoder Iteratively
        for i in range(self.decoding_sequence_length - 1):
            decoder_inputs = {
                'decoder_input_ids': decoder_input_ids,       # Token IDs (int64)
                'encoder_hidden_states': encoded_features  # Encoder output (float32)
            }

            decoder_outputs = self.decoder_session.run(None, decoder_inputs)

            repetition_penalty = 1.5
            logits = apply_repetition_penalty(decoder_outputs[0][:, i], generated_tokens, penalty=repetition_penalty, last_window=4)
            next_token = np.argmax(logits)  # greedy decoding

            generated_tokens.append(next_token)

            decoder_input_ids[0][i + 1] = np.array([[next_token]], dtype=np.int64)  # Shape (1, seq_len+1)

            if next_token == self.tokenizer.eos_token_id:
                break

        # Convert token IDs to text
        transcription = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        transcriptions.append(transcription)

        return transcriptions
