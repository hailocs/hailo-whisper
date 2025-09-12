# Model Conversion

This document explains the necessary step to successfully modify, export and convert a Whisper tiny and base models for Hailo-8 and Hailo10H.


## Before starting with the model conversion
You may notice that the **setup.py** script run during the installation process populated the OpenAI's Whisper submodule in the *third_party* folder.  
In addition, the setup script also applied some modifications to the PyTorch source code to optimize the mode for the Hailo-8 architecture, and make it compatible with the operations supported by the Hailo SDK.
You can inspect the applied changes in the *hailo-compatibility-changes.patch* file.


## Export the model
The *export* folder contains a few scripts to export different variants of the Whisper model to ONNX.
To export the model, run the following from the repository root folder:
```
python3 -m export.export_whisper_model --variant WHISPER_VARIANT
```

where *WHISPER_VARIANT* can be *tiny* or *base*.  
**Note**: the export script will apply some modifications to the PyTorch code. For the *base* variant, the script decreases the input size to 5 seconds, otherwise the model will not compile. This is done by changing the following line in *third_party/whisper/whisper/model.py*:
*Whisper-tiny*
```python
self.dims.n_audio_ctx // 3,  # 10 seconds input. Only for whisper tiny
```

*Whisper-base*
```python
self.dims.n_audio_ctx // 6,  # 5 seconds input. For whisper base (even tiny can support 5 seconds, of course).
```
This is done automatically, but it is important to keep it in mind when proceeding with the model conversion.

## Encoder conversion

The whisper encoder conversion script will also create the calibration set for you. In order to do this, the dataset must be available. In this example, the [LibriSpeech](https://www.openslr.org/12) dataset will be used, since it is one of the dataset the Whisper model was trained on.

### Download the dataset

The [dev-clean](https://www.openslr.org/resources/12/dev-clean.tar.gz) **LibriSpeech dataset is download by the setup script**.
If not available, please proceed with the manual download as below.


- Download the [dev-clean](https://www.openslr.org/resources/12/dev-clean.tar.gz) dataset.
- Create an *audio* folder in the repository's root folder.
- Extract the archive inside the *audio* folder.
- The *audio* directory must look like this:
  ```
   audio/
   ├── dev-clean
   │   └── LibriSpeech
   │       ├── BOOKS.TXT
   │       ├── CHAPTERS.TXT
   │       ├── dev-clean
   │       │   ├── 1272
           ...
           ...
  ```

> **NOTE:** when using a custom dataset, please create a new method in *convert_whisper_encoder.py* to parse your dataset structure and generate the Mel spectrogram, as done in the *create_calibration_set_librispeech* function.

### Convert the encoder

To convert the Whisper-tiny encoder for Hailo, please run the following command from the main folder:

```
python3 -m conversion.convert_whisper_encoder ONNX_ENCODER_PATH --variant WHISPER_VARIANT --hw-arch HAILO_ARCH
```

Please explore also the command line arguments:
```
usage: convert_whisper_encoder.py [-h] [--load-calib-set] --variant {tiny,tiny.en,base,base.en} [--hw-arch {hailo8,hailo8l,hailo10h}] model_path

Whisper encoder conversion script for Hailo

positional arguments:
  model_path            Encoder ONNX model path

options:
  -h, --help            show this help message and exit
  --load-calib-set      Reuse the previously generated calibration set (default: False)
  --variant {tiny,tiny.en,base,base.en}
                        Whisper model variant to convert
  --hw-arch {hailo8,hailo8l,hailo10h}
                        Hardware architecture to use (default: hailo8)

```

## Decoder conversion

The Whisper decoder conversion is handled thorugh two separate scripts:
- *create_decoder_calib_set.py*: generates the decoder calibration set
- *convert_whisper_decoder.py* converts the whisper model using the Hailo DFC

### Create the decoder calibration set

Creating the calibration set for the decoder is a 3-step process:


1. Run the encoder on the calibration set to get the encoded features.
2. Run the decoder on the encoded features to get the sequence of tokens.
3. Run the tokenization on the full sequence of tokens to get the intermediate input.


Please run the command below to create the calibration set

```
python3 -m conversion.create_decoder_calib_set --encoder ONNX_ENCODER_PATH --decoder ONNX_DECODER_PATH --variant WHISPER_VARIANT
```

For example:
```
python3 -m conversion.create_decoder_calib_set --encoder export/tiny-whisper-encoder-10s.onnx --decoder export/tiny-whisper-decoder-10s-seq-32.onnx --variant tiny
```
### Convert the decoder

After creating the dataset, you can proceed with the model conversion:

```
python3 -m conversion.convert_whisper_decoder ONNX_DECODER_PATH --variant WHISPER_VARIANT --hw-arch HAILO_ARCH
```

For example:
```
python3 -m conversion.convert_whisper_decoder export/tiny-whisper-decoder-10s-seq-32.onnx --variant tiny --hw-arch hailo8
```

Please use the helper to check the command line arguments:
```
usage: convert_whisper_decoder.py [-h] [--variant {tiny,tiny.en,base,base.en}] [--hw-arch {hailo8,hailo8l,hailo10h}] model_path

Whisper decoder conversion script for Hailo

positional arguments:
  model_path            Decoder ONNX model path

options:
  -h, --help            show this help message and exit
  --variant {tiny,tiny.en,base,base.en}
                        Whisper model variant to convert
  --hw-arch {hailo8,hailo8l,hailo10h}
                        Hardware architecture to use (default: hailo8)

```


## Model Evaluation
Please refer to the [Evaluation](./Evaluation.md) documentation for instructions to run tests and evaluation using either an ONNX or HAR file.


## Integration in Application Example

It is possible to integrate the converted HEF files into our [Speech Recognition example](https://github.com/hailo-ai/Hailo-Application-Code-Examples/blob/main/runtime/hailo-8/python/speech_recognition/README.md) application.  
Since the embedding operators have been removed from the model at conversion time, they must run on the host CPU. The npy files containing the parameters for these operators are generated under the *conversion/optimization/* folder when the *create_decoder_calib_set.py* script is run. These files - and the converted HEFs - must be copied into the application example, and the application code must be modified to point to the new files.


## Disclaimer
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
