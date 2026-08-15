"""
==============================================================================
FILE: test_hf.py
ROLE: Hugging Face Authentication & Environment Sanity Test Script
BRANCH ADDITION (abhyuday): Newly introduced simple script for validating Hugging Face
Hub authentication and connectivity before launching model downloads or training.
==============================================================================
"""

from huggingface_hub import login
print("Testing HF access... please enter your token if prompted.")
