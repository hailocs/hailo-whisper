from abc import ABC, abstractmethod


class WhisperEncoder(ABC):

    def __init__(self, encoder_model_path: str):
        """
        Initialize the model path.
        :param encoder_model_path: Path to the encoder model file.
        """
        self.encoder_model_path = encoder_model_path
        self.backend = None
        self.input_audio_length = None

    @abstractmethod
    def encode(self, input_mel):
        """Encodes the input mel spectrogram and returns latent representations."""
        pass

    def get_input_length(self):
        return self.input_audio_length


class WhisperDecoder(ABC):

    def __init__(self, decoder_model_path: str):
        """
        Initialize the model path.
        :param decoder_model_path: Path to the decoder model file.
        """
        self.decoder_model_path = decoder_model_path
        self.backend = None
        self.decoding_sequence_length = None

    @abstractmethod
    def decode(self, encoded_features):
        """Decodes the latent features into text using prompt tokens."""
        pass
    