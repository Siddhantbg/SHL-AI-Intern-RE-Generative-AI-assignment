#!/usr/bin/env python3
"""
Simple script to generate submission CSV with firstname_lastname format.
"""

import csv
import sys
import shutil
from pathlib import Path

def create_submission_csv(firstname, lastname):
    """Create submission CSV by copying and renaming the test predictions."""
    
    # Source file with your test predictions
    source_file = Path("test_predictions_data.csv")
    
    if not source_file.exists():
        print(f"Error: Source file {source_file} not found!")
        return None
    
    # Create output directory
    output_dir = Path("submission")
    output_dir.mkdir(exist_ok=True)
    
    # Create CSV filename with firstname_lastname format
    csv_filename = f"{firstname}_{lastname}.csv"
    csv_path = output_dir / csv_filename
    
    # Copy the source file to the new location with the correct name
    shutil.copy2(source_file, csv_path)
    
    # Count rows for verification
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
        total_rows = len(rows) - 1  # Subtract header row
    
    print(f"✅ Created submission CSV: {csv_path}")
    print(f"📊 Total predictions: {total_rows}")
    print(f"📁 File size: {csv_path.stat().st_size} bytes")
    
    # Show first few rows for verification
    print("\n📋 First few rows:")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i < 3:  # Show header + first 2 data rows
                print(f"  {row}")
            else:
                break
    
    return csv_path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_submission.py <firstname> <lastname>")
        print("Example: python generate_submission.py john doe")
        print("\nThis will create a file named 'john_doe.csv' in the submission/ directory")
        sys.exit(1)
    
    firstname = sys.argv[1].lower()
    lastname = sys.argv[2].lower()
    
    result = create_submission_csv(firstname, lastname)
    
    if result:
        print(f"\n🎉 Success! Your submission file is ready: {result}")
        print("You can now submit this CSV file with your predictions.")
    else:
        print("❌ Failed to create submission file.")
        sys.exit(1)