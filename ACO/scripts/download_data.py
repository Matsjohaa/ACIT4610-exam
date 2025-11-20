"""
Download OR-Library bin packing benchmark instances
"""
import os
import requests
from pathlib import Path

# Base URL for OR-Library
BASE_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"

# Benchmark files to download
BENCHMARKS = [
    "binpack1.txt",
    "binpack2.txt", 
    "binpack3.txt",
    "binpack4.txt",
    "binpack5.txt",
    "binpack6.txt",
    "binpack7.txt",
    "binpack8.txt",
]

def download_data(output_dir: str = "aco/data/raw"):
    """Download all OR-Library bin packing instances"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading OR-Library bin packing benchmarks to {output_dir}/\n")
    
    for filename in BENCHMARKS:
        url = BASE_URL + filename
        output_path = os.path.join(output_dir, filename)
        
        # Skip if already exists
        if os.path.exists(output_path):
            print(f"⏭️  {filename} already exists, skipping...")
            continue
        
        try:
            print(f"⬇️  Downloading {filename}...", end=" ")
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise error for bad status
            
            # Save to file
            with open(output_path, 'w') as f:
                f.write(response.text)
            
            print("✅")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed: {e}")
    
    print(f"\n✅ Download complete! Files saved to {output_dir}/")
    
    # Print summary
    print("\n📊 Downloaded benchmarks:")
    for filename in BENCHMARKS:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                first_line = f.readline().strip().split()
                n, C, best = first_line[0], first_line[1], first_line[2] if len(first_line) > 2 else "?"
            print(f"  - {filename}: n={n}, C={C}, best_known={best}")


if __name__ == "__main__":
    download_data()