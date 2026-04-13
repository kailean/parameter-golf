#!/usr/bin/env python3
"""Download Parameter Golf dataset to external SSD"""
import os
import sys

# Set paths to external SSD
EXTERNAL_DATA = "/Volumes/MacStorageExtended/parameter-golf-data"
os.makedirs(EXTERNAL_DATA, exist_ok=True)

# Create symlink from parameter-golf/data to external
data_link = "/Users/kaileanhard/.openclaw/workspace/parameter-golf/data/datasets"
tokenizer_link = "/Users/kaileanhard/.openclaw/workspace/parameter-golf/data/tokenizers"

# Remove existing directories if they exist
import shutil
if os.path.islink(data_link):
    os.unlink(data_link)
elif os.path.isdir(data_link):
    shutil.rmtree(data_link)
    
if os.path.islink(tokenizer_link):
    os.unlink(tokenizer_link)
elif os.path.isdir(tokenizer_link):
    shutil.rmtree(tokenizer_link)

# Create directories on external
os.makedirs(f"{EXTERNAL_DATA}/datasets", exist_ok=True)
os.makedirs(f"{EXTERNAL_DATA}/tokenizers", exist_ok=True)

# Create symlinks
os.symlink(f"{EXTERNAL_DATA}/datasets", data_link)
os.symlink(f"{EXTERNAL_DATA}/tokenizers", tokenizer_link)

print(f"✅ Data directories linked to external SSD")
print(f"   Location: {EXTERNAL_DATA}")

# Now run the download script
print("\n🚀 Starting dataset download...")
os.chdir("/Users/kaileanhard/.openclaw/workspace/parameter-golf/data")

# Set environment variable for the script
os.environ["MATCHED_FINEWEB_REPO_ID"] = "willdepueoai/parameter-golf"

# Import and run the download
import subprocess
result = subprocess.run([
    sys.executable, "cached_challenge_fineweb.py",
    "--train-shards", "10",
    "--variant", "sp1024"
], capture_output=False, text=True)

sys.exit(result.returncode)