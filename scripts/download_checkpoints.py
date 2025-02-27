#!/usr/bin/env python3
# Located in experiments/scripts/download_checkpoints.py
"""
Script to download model checkpoints from S3 bucket to a local directory
and set the appropriate environment variable.

Usage:
    python download_checkpoints.py [--output-dir DIR]

Options:
    --output-dir DIR   Directory to store the model checkpoints
                       Default: ./pretrained_models/
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Download model checkpoints from S3')
    parser.add_argument('--output-dir', type=str, default='./pretrained_models/',
                        help='Directory to store the model checkpoints (default: ./pretrained_models/)')
    return parser.parse_args()

def check_aws_cli():
    """Check if AWS CLI is installed and configured."""
    try:
        subprocess.run(['aws', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def download_checkpoints(output_dir):
    """Download model checkpoints from S3 bucket."""
    s3_bucket = 's3://pleas-merging-artiacts-sewoonglab'
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model checkpoints from {s3_bucket} to {output_dir}...")
    
    # Use AWS CLI to download
    try:
        subprocess.run([
            'aws', 's3', 'sync', 
            s3_bucket, 
            output_dir,
            '--no-sign-request'  # Use this flag if the bucket allows public access
        ], check=True)
        print("Download completed successfully!")
        return True
    except subprocess.SubprocessError as e:
        print(f"Error downloading from S3: {e}")
        return False

def set_environment_variable(output_dir):
    """Set environment variable to point to the model directory."""
    # Get absolute path
    abs_path = os.path.abspath(output_dir)
    
    # Set environment variable for current process
    os.environ['MODEL_CHECKPOINTS_DIR'] = abs_path
    print(f"Environment variable MODEL_CHECKPOINTS_DIR set to: {abs_path}")
    
    # Provide commands for users to set it permanently
    if os.name == 'posix':  # Linux/Mac
        print("\nTo set this environment variable permanently, run:")
        print(f"echo 'export MODEL_CHECKPOINTS_DIR=\"{abs_path}\"' >> ~/.bashrc")
        print("source ~/.bashrc")
    elif os.name == 'nt':  # Windows
        print("\nTo set this environment variable permanently, run:")
        print(f"setx MODEL_CHECKPOINTS_DIR \"{abs_path}\"")
    
    # Return the path for the calling script to use
    return abs_path

def main():
    args = parse_args()
    output_dir = args.output_dir
    
    # Check if AWS CLI is installed
    if not check_aws_cli():
        print("AWS CLI is not installed or not in PATH. Please install it first:")
        print("https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html")
        sys.exit(1)
    
    # Download checkpoints
    if download_checkpoints(output_dir):
        # Set environment variable
        checkpoint_dir = set_environment_variable(output_dir)
        print(f"\nModel checkpoints are ready to use at: {checkpoint_dir}")
    else:
        print("\nFailed to download checkpoints. Please check your internet connection and AWS permissions.")
        sys.exit(1)

if __name__ == "__main__":
    main()