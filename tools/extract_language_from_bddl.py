#!/usr/bin/env python3
"""
Extract language instructions from BDDL files and generate a JSON mapping.

This script reads all .bddl files containing "language" in their filename
from the libero_goal directory, extracts the (:language) value from each file,
and creates a JSON file mapping language text to the task name (filename prefix
before "_language").
"""

import json
import re
from pathlib import Path


def extract_language_from_bddl(file_path):
    """Extract the (:language) value from a BDDL file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Match (:language ...) pattern
        # The pattern matches (:language followed by text until the closing parenthesis
        match = re.search(r'\(:language\s+(.+?)\)', content, re.DOTALL)
        if match:
            language_text = match.group(1).strip()
            return language_text
        else:
            print(f"Warning: Could not find (:language) in {file_path}")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def extract_task_name_from_filename(filename):
    """Extract the task name (prefix before '_language') from filename."""
    # Remove .bddl extension
    name_without_ext = filename.replace('.bddl', '')
    # Find the position of '_language'
    if '_language' in name_without_ext:
        task_name = name_without_ext.split('_language')[0]
        # Replace underscores with spaces
        task_name = task_name.replace('_', ' ')
        return task_name
    else:
        print(f"Warning: Could not find '_language' in filename {filename}")
        return None


def main():
    # Directory containing BDDL files
    bddl_dir = Path('/home/ubuntu/Desktop/hld/openpi/third_party/LIBERO-plus/libero/libero/bddl_files/libero_goal')
    
    # Output JSON file
    output_file = Path('/home/ubuntu/Desktop/hld/openpi/libero_goal_language_mapping.json')
    
    # Dictionary to store language -> task_name mapping
    language_mapping = {}
    
    # Find all .bddl files containing "language" in filename
    bddl_files = list(bddl_dir.glob('*language*.bddl'))
    
    print(f"Found {len(bddl_files)} BDDL files containing 'language' in filename")
    
    # Process each file
    for bddl_file in sorted(bddl_files):
        # Extract language text from file
        language_text = extract_language_from_bddl(bddl_file)
        if language_text is None:
            continue
        
        # Extract task name from filename
        task_name = extract_task_name_from_filename(bddl_file.name)
        if task_name is None:
            continue
        
        # Store in mapping (if same language appears multiple times, last one wins)
        language_mapping[language_text] = task_name
        
        print(f"Processed: {bddl_file.name} -> {task_name}")
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(language_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated JSON file: {output_file}")
    print(f"Total mappings: {len(language_mapping)}")


if __name__ == '__main__':
    main()

