#!/bin/bash
# This script downloads the LibriSpeech dataset for speech recognition tasks.

wget https://www.openslr.org/resources/12/dev-clean.tar.gz
mkdir dev-clean
tar -xzvf dev-clean.tar.gz -C dev-clean/
rm dev-clean.tar.gz