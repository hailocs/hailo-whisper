# Hailo Whisper - Whisper Model Conversion and Evaluation for Hailo Devices

This repository provides a streamlined approach to convert and deploy OpenAI's Whisper model on the Hailo-8 and Hailo-10H AI accelerators. It includes necessary scripts and configurations to export, convert and evaluate the models using Hailo SDK.  

## Supported models
- Whisper *tiny* / *tiny.en*
- Whisper *base* / *base.en*

## Prerequisites

Ensure your system matches the following requirements before proceeding:

- Platforms tested: x86
- OS: Ubuntu 22 (x86)
- (Optional) **HailoRT 4.20 or later** and the corresponding **PCIe driver** must be installed. You can download them from the [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- **ffmpeg** and **libportaudio2** installed for audio processing.
  ```
  sudo apt update
  sudo apt install ffmpeg
  sudo apt install libportaudio2
  ```
- **Python 3.10 or 3.11** installed.

## Installation

Follow these steps to set up the environment and install dependencies for inference:

1. Clone this repository:

   ```sh
   git clone https://github.com/hailocs/hailo-whisper.git
   cd hailo-whisper
   ```

2. Run the setup script to install dependencies:  

   ```sh
   python3 setup.py
   ```

3. Activate the virtual environment from the repository root folder:

   ```sh
   source whisper_env/bin/activate
   ```

4. Download and install the Hailo Dataflow Compiler (at least v3.31.0) in the virtual environment. **Use DFC v3.x for Hailo-8/8L, or DFC 5.x for Hailo-10H**.

5. (Optional) Install PyHailoRT inside the virtual environment (must be downloaded from the Hailo Developer Zone), for example:
   ```sh
   pip install hailort-4.21.0-cp310-cp310-linux_x86_64.whl
   ```
   PyHailoRT is version must match the installed HailoRT version. PyHailoRT is useful to test the converted model if you have an Hailo-8 module connected to the PC used for development.


## Model conversion

Once the installation is completed, please proceed with the [Model Conversion](docs/Conversion.md).


## Application Example

For instructions to install and run an app using converted HEF files, please refer to the [Speech Recognition example](https://github.com/hailo-ai/Hailo-Application-Code-Examples/blob/main/runtime/hailo-8/python/speech_recognition/README.md).


## Disclaimer
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This example was tested on specific versions and we can only guarantee the expected results using the exact version mentioned above on the exact environment. The example might work for other versions, other environment or other HEF file, but there is no guarantee that it will.
