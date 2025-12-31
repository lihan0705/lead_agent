#!/usr/bin/env python3
"""
Download GAIA benchmark dataset
"""

import os
from datasets import load_dataset
from huggingface_hub import snapshot_download


def download_gaia_dataset():
    """Download GAIA dataset from HuggingFace"""
    
    print("🚀 Starting GAIA dataset download...")
    
    # Download the dataset
    print("📥 Downloading dataset from HuggingFace...")
    data_dir = snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset")
    print(f"✅ Dataset downloaded to: {data_dir}")
    
    # Load the test split for 2023 level 1
    print("\n📊 Loading 2023_level1 test split...")
    dataset = load_dataset(data_dir, "2023_level1", split="test")
    print(f"✅ Loaded {len(dataset)} examples from 2023_level1/test")
    
    # Display some examples
    print("\n📋 Sample examples:")
    print("=" * 80)
    
    for i, example in enumerate(dataset[:3]):  # Show first 3 examples
        print(f"\n--- Example {i + 1} ---")
        print(f"Question: {example['Question']}")
        
        # Check if file_path exists
        if 'file_path' in example and example['file_path']:
            file_path = os.path.join(data_dir, example['file_path'])
            print(f"File path: {file_path}")
            print(f"File exists: {os.path.exists(file_path)}")
        
        # Show other fields
        for key in example.keys():
            if key not in ['Question', 'file_path']:
                print(f"{key}: {example[key]}")
        
        print("-" * 80)
    
    # Save dataset info
    info_file = "/Users/lihan/Documents/lead_agent/gaia_dataset_info.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"GAIA Dataset Information\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Split: 2023_level1/test\n")
        f.write(f"Number of examples: {len(dataset)}\n\n")
        f.write(f"Dataset features:\n")
        for feature in dataset.features:
            f.write(f"  - {feature}: {dataset.features[feature]}\n")
    
    print(f"\n💾 Dataset info saved to: {info_file}")
    print(f"\n✨ Download complete!")
    print(f"\nYou can now use the dataset with:")
    print(f"  from datasets import load_dataset")
    print(f"  dataset = load_dataset('{data_dir}', '2023_level1', split='test')")
    
    return data_dir, dataset


if __name__ == "__main__":
    try:
        data_dir, dataset = download_gaia_dataset()
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        raise
