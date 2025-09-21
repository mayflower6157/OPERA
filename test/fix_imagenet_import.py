#!/usr/bin/env python3
"""
Fix the ImageNetInfo import error in transformers gemma3n configuration.
"""

import sys
import re

def fix_import_error(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the problematic import with a try-except block
    old_import = "if is_timm_available():\n    from timm.data import ImageNetInfo, infer_imagenet_subset"
    
    new_import = """if is_timm_available():
    try:
        from timm.data import ImageNetInfo, infer_imagenet_subset
    except ImportError:
        # Fallback for timm versions that don't have ImageNetInfo
        ImageNetInfo = None
        def infer_imagenet_subset(config_dict):
            return None"""
    
    content = content.replace(old_import, new_import)
    
    # Also update the code that uses these imports to handle None case
    content = re.sub(
        r'if imagenet_subset:',
        'if imagenet_subset and ImageNetInfo is not None:',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed import error in {file_path}")

if __name__ == "__main__":
    fix_import_error("/home/ubuntu/miniconda/envs/opera/lib/python3.10/site-packages/transformers/models/gemma3n/configuration_gemma3n.py")
