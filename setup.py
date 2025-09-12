import os
import subprocess
import sys
import argparse

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
VENV_DIR = os.path.join(ROOT_DIR, "whisper_env")
PYTHON_BIN = os.path.join(VENV_DIR, "bin", "python")
PIP_BIN = os.path.join(VENV_DIR, "bin", "pip")
WHISPER_DIR = os.path.join(ROOT_DIR, "third_party", "whisper")
DATASET_DIR = os.path.join(ROOT_DIR, "audio")
PATCH_FILE = os.path.join(ROOT_DIR, "hailo-compatibility-changes.patch")

def run_command(command, cwd=None):
    """Helper function to run shell commands."""
    subprocess.run(command, shell=True, cwd=cwd, check=True)

def create_venv():
    """Creates a virtual environment if it doesn't exist."""
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment in {VENV_DIR}...")
        run_command(f"python3 -m venv {VENV_DIR}")
    else:
        print("Virtual environment already exists.")

    # Upgrade pip to the latest version
    print("\nUpgrading pip inside the virtual environment...")
    run_command(f"{PIP_BIN} install --upgrade pip")

def setup_whisper_submodule():
    """Initializes and updates the Whisper submodule, then applies the patch."""
    if not os.path.exists(WHISPER_DIR) or not os.listdir(WHISPER_DIR):
        print(f"Initializing and updating the Whisper submodule in {WHISPER_DIR}...")
        run_command("git submodule init", cwd=ROOT_DIR)
        run_command("git submodule update", cwd=ROOT_DIR)
    else:
        print("Whisper submodule already exists.")

    print("Applying Hailo compatibility patch...")
    try:
        run_command(f"git apply {PATCH_FILE}", cwd=WHISPER_DIR)
        print("Patch applied successfully.")
    except subprocess.CalledProcessError:
        print("Patch application failed. It may already be applied or not compatible.")

    print("Installing Whisper module...")
    try:
        run_command(f"{PIP_BIN} install --no-deps -e .", cwd=WHISPER_DIR)  # skip dependencies as they are handled separately
        print("Whisper sources installed successfully.")
    except subprocess.CalledProcessError:
        print("Whisper installation failed. Is it already installed?")

def download_dataset():
    """Downloads the dataset if it doesn't exist."""
    clean_dataset_path = os.path.join(DATASET_DIR, "dev-clean")
    if not os.path.exists(clean_dataset_path):
        print("Downloading dataset...")
        run_command(f"./download_dataset.sh", cwd=DATASET_DIR)
        print("Dataset downloaded successfully.")
    else:
        print("Dataset already exists, skipping download.")

def install_requirements(develop_install=True):
    """Installs required Python packages inside the virtual environment."""
    if develop_install:
        requirements_develop_file = os.path.join(ROOT_DIR, "requirements.txt")
        if os.path.exists(requirements_develop_file):
            print("\nInstalling dependencies from requirements.txt...")
            run_command(f"{PIP_BIN} install -r {requirements_develop_file}")
        else:
            print("No requirements.txt found, skipping package installation.")
        setup_whisper_submodule()
        download_dataset()
    

def main():
    """Main function to set up the environment."""

    create_venv()
    install_requirements(develop_install=True)

    print("\n✅ Setup complete! To activate the environment, run:")
    print(f"source {VENV_DIR}/bin/activate\n")

if __name__ == "__main__":
    main()
