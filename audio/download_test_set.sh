#!/bin/bash
# This script downloads the test LibriSpeech dataset for evaluation purposes.

wget https://www.openslr.org/resources/12/test-clean.tar.gz
mkdir test-clean
tar -xzvf test-clean.tar.gz -C test-clean/
rm test-clean.tar.gz