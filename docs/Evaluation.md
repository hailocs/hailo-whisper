# Evaluation with ONNX & Hailo SDK

This document explains the scripts to evaluate OpenAI’s Whisper model using either ONNX Runtime or the Hailo SDK.
Two scripts are available in the **evaluation** folder, *testbench.py* and *evaluation.py*:  

- **testbench.py** allows the user to run inference on a single audio file, useful for a quick accuracy evaluation
- **evaluation.py** allows the user to run a proper evaluation on an arbitrary amount of input audio files, calculatin the Word Error Rate (WER).

The framework is flexible: you can run the encoder and decoder independently with either backend, making it possible to mix and match to identify potential accuracy issues.
In addition, Hailo SDK supports different targets for the execution:
- *native*, the model (HAR) is run in FP32. This target is available for every HAR file, even before quantization
- *quantized*, the model is run in INT format, according to the precision selected during the quantization. It requires a quantized HAR file
- "hw", the model is run on the Hailo device. It requires a compiled HAR file, and the HailoRT and driver must be installed.


## Testbench
Use the testbench to run inference on a single audio file.

```
usage: testbench.py [-h] --variant {tiny,tiny.en,base,base.en} --encoder ENCODER --decoder DECODER [--encoder-target {native,quantized,hw}] [--decoder-target {native,quantized,hw}] [--audio-path AUDIO_PATH]

Encoder/Decoder inference emulation script

options:
  -h, --help            show this help message and exit
  --variant {tiny,tiny.en,base,base.en}
                        Whisper model variant to run
  --encoder ENCODER     Whisper encoder path (ONNX / HAR)
  --decoder DECODER     Whisper decoder path (ONNX / HAR)
  --encoder-target {native,quantized,hw}
                        Optional target for encoder
  --decoder-target {native,quantized,hw}
                        Optional target for decoder
  --audio-path AUDIO_PATH
                      Audio file path for the test (if None, uses a default file)

```

#### Examples
- ONNX encoder - ONNX decoder, using default audio file:
  ```
  python3 -m evaluation.testbench --encoder export/tiny-whisper-encoder-10s.onnx --decoder export/tiny-whisper-decoder-10s-seq-32.onnx --variant tiny
  ```

- Hailo quantized encoder - ONNX decoder, using specific audio file:
  ```
  python3 -m evaluation.testbench --encoder conversion/converted/tiny_whisper_encoder_10s_hailo8/tiny-whisper-encoder-10s.har \
                                  --encoder-target quantized \
                                  --decoder export/tiny-whisper-decoder-10s-seq-32.onnx \
                                  --variant tiny \
                                  --audio-path audio/dev-clean/LibriSpeech/dev-clean/3752/4943/3752-4943-0000.flac
  ```

- Hailo quantized encoder - Hailo quantized decoder:
  ```
  python3 -m evaluation.testbench --encoder conversion/converted/tiny_whisper_encoder_10s_hailo8/tiny-whisper-encoder-10s.har \
                                  --encoder-target quantized \
                                  --decoder conversion/converted/tiny_whisper_decoder_10s_seq_32_hailo8/tiny-whisper-decoder-10s-seq-32.har \
                                  --decoder-target \
                                  --variant tiny

  ```

## Evaluation

The same flags used in testbench,py are valid also for evaluation, with the addition of *--num-samples* argument to specify the amount of input audio files the evaluation should be run on.

```
usage: evaluation.py [-h] --variant {tiny,tiny.en,base,base.en} --encoder ENCODER --decoder DECODER [--encoder-target {native,quantized,hw}] [--decoder-target {native,quantized,hw}] [--num-samples NUM_SAMPLES]

Encoder/Decoder inference evaluation

options:
  -h, --help            show this help message and exit
  --variant {tiny,tiny.en,base,base.en}
                        Whisper model variant to run
  --encoder ENCODER     Whisper encoder path (ONNX / HAR)
  --decoder DECODER     Whisper decoder path (ONNX / HAR)
  --encoder-target {native,quantized,hw}
                        Optional target for encoder
  --decoder-target {native,quantized,hw}
                        Optional target for decoder
  --num-samples NUM_SAMPLES
                        Number of samples to run evaluation on (default: 50)

```

#### Examples

- Hailo quantized encoder - Hailo quantized decoder
  ```
  python3 -m evaluation.evaluation --encoder conversion/converted/tiny_whisper_encoder_10s_hailo8/tiny-whisper-encoder-10s.har \
                                  --encoder-target quantized \
                                  --decoder conversion/converted/tiny_whisper_decoder_10s_seq_32_hailo8/tiny-whisper-decoder-10s-seq-32.har \
                                  --decoder-target \
                                  --variant tiny \
                                  --num-samples 100

  ```
