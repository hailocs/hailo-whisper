import numpy as np
from evaluation.base.whisper_runner import WhisperDecoder
from hailo_sdk_client import ClientRunner, InferenceContext
from transformers import AutoTokenizer
from common.postprocessing import apply_repetition_penalty, temperature_sampling
import os
import sys


class HailoSdkWhisperDecoder(WhisperDecoder):

    def __init__(self, decoder_model_path: str, variant="tiny", target="native"):
        """
        :param decoder_model_path: Path to the encoder model file.
        """
        super().__init__(decoder_model_path)
        self.decoder_runner = ClientRunner(har=self.decoder_model_path)
        self.decoder_runner.model_name
        self.backend = "hailo"
        self.target = target
        self.variant = variant
        self.decoding_sequence_length = self._get_decoding_sequence_length()
        self.decoder_model_name = self.decoder_runner.model_name

        # token embedding
        self.token_embedding_weight = self._load_token_embedding_weight()
        self.onnx_add_input = self._load_onnx_add_input()
        self.constant_output_0 = np.array([1])  # Unsqueeze axis
        self._load_tokenizer()

    def _get_decoding_sequence_length(self):
        layers = self.decoder_runner.get_hn()["layers"]
        for layer_name in layers:
            if "input_layer2" in layer_name:
                decoding_sequence_length = layers[layer_name]["input_shapes"][0][1]
                break
        return decoding_sequence_length

    def _load_token_embedding_weight(self):
        token_embedding_path = f"./conversion/optimization/{self.variant}/token_embedding_weight_{self.variant}_seq_{self.decoding_sequence_length}.npy"
        if not os.path.exists(token_embedding_path):
            print(f"Tokenization assets not found for {self.variant}. Please run the tokenization script to extract them.")
            print(f"python3 -m conversion.tokenization ONNX_DECODER_PATH --variant {self.variant}")
            sys.exit()
        token_embedding = np.load(token_embedding_path)
        return token_embedding

    def _load_onnx_add_input(self):
        onnx_add_input = np.load(f"./conversion/optimization/{self.variant}/onnx_add_input_{self.variant}_seq_{self.decoding_sequence_length}.npy")
        return onnx_add_input

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai/whisper-" + self.variant)

    def tokenization(self, decoder_input_ids):
        # 1. Gather operation (embedding lookup)
        gather_output = self.token_embedding_weight[decoder_input_ids]  # Shape: (len(decoder_input_ids), 384)
        # 2. Add bias
        add_output = gather_output + self.onnx_add_input  # Broadcasting with shape (32, 384)
        # 3. Unsqueeze (insert dimension at axis=1)
        unsqueeze_output = np.expand_dims(add_output, axis=int(self.constant_output_0[0]))  # Shape: (32, 1, 384)
        # 4. Transpose
        transpose_output = np.transpose(unsqueeze_output, (0, 3, 2, 1))  # Reordering dimensions

        return transpose_output
    
    def apply_matmul(self, last_normalization_output):
        matmul_output = np.matmul(np.squeeze(last_normalization_output), self.matmul_params)
        matmul_output = np.expand_dims(matmul_output, axis=0)
        # print(matmul_output.shape)
        return matmul_output

    def decode(self, encoded_features):

        transcriptions = []
        # print(encoded_features.shape)
        if ((len(encoded_features.shape) == 3) and (self.target != "hw")):   # i.e. if the encoder was executed on the HW bnut the decoder is not
            encoded_features = np.expand_dims(encoded_features, axis=0)  # compatibility with emulator

        # Assuming <|startoftranscript|> is the start-of-sequence token, which should be encoded as a scalar

        # start_token_id = self.tokenizer.encode('<|startoftranscript|>', truncation=True, padding=False, add_special_tokens=False)
        start_token_id = [50258]

        decoder_input_ids = np.array([[start_token_id[0]]], dtype=np.int64)  # Shape (1,1)
        decoder_input_ids = np.concatenate([decoder_input_ids, np.zeros((1, self.decoding_sequence_length - 1), dtype=np.int64)], axis=1)

        generated_tokens = []

        matmul_on_host = False
        concat_on_host = True
        decoder_outputs = None
        # Run Decoder Iteratively
        for i in range(self.decoding_sequence_length - 1):

            tokenized_ids = self.tokenization(decoder_input_ids)
            tokenized_ids = np.transpose(tokenized_ids, [0, 2, 3, 1])  # NHWC. TODO: merge this in self.tokenization
            # print(tokenized_ids.shape)

            decoder_inputs = {
                self.decoder_model_name.replace(".", "_") + '/input_layer1': encoded_features,
                self.decoder_model_name.replace(".", "_") + '/input_layer2': tokenized_ids
            }
            if self.target == "quantized":
                with self.decoder_runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
                    decoder_outputs = self.decoder_runner.infer(ctx, decoder_inputs)
            elif self.target == "hw":
                with self.decoder_runner.infer_context(InferenceContext.SDK_HAILO_HW) as ctx:
                    decoder_outputs = self.decoder_runner.infer(ctx, decoder_inputs)
            else:
                with self.decoder_runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
                    decoder_outputs = self.decoder_runner.infer(ctx, decoder_inputs)

            if matmul_on_host:
                decoder_outputs = self.apply_matmul(decoder_outputs)
                # print(decoder_outputs.shape)

                repetition_penalty = 1.5  # Adjust as needed
                logits = apply_repetition_penalty(decoder_outputs[:, i], generated_tokens, penalty=repetition_penalty)
                if self.variant == "tiny" or self.variant == "base":
                    next_token = np.argmax(logits)  # greedy decoding
                else:
                    next_token = temperature_sampling(logits, temperature=0.3)

            elif concat_on_host:

                decoder_outputs = np.concatenate((decoder_outputs[0], decoder_outputs[1], decoder_outputs[2], decoder_outputs[3]), axis=3)
                decoder_outputs = np.squeeze(decoder_outputs, axis=0)

                repetition_penalty = 1.5
                logits = apply_repetition_penalty(decoder_outputs[:, i], generated_tokens, penalty=repetition_penalty, last_window=4)
                next_token = np.argmax(logits)  # greedy decoding
            else:
                next_token = np.argmax(decoder_outputs[0][:, i])  # Get most probable token. Look just at the i index in the second dimension (current prediction)
            # print(next_token)
            generated_tokens.append(next_token)
            
            decoder_input_ids[0][i + 1] = np.array([[next_token]], dtype=np.int64)  # Shape (1, seq_len+1)

            if next_token == self.tokenizer.eos_token_id:
                break

        # Convert token IDs to text
        # print(generated_tokens)
        transcription = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        transcriptions.append(transcription)

        return transcriptions
