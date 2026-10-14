import json
import os
import re
import json
import shutil

def replace_multiple_patterns(notebook_path, replacements_dict, output_path=None):
    """
    Read a Jupyter notebook file and replace multiple patterns based on a dictionary
    
    Parameters:
    -----------
    notebook_path : str
        Path to the notebook file
    replacements_dict : dict
        Dictionary with old_pattern: new_pattern pairs
    output_path : str, optional
        Path to save the modified notebook. If None, overwrites the original
    
    Returns:
    --------
    dict : Dictionary with status of each replacement
    """
    
    # Read the notebook file
    with open(notebook_path, 'rb') as f:
        content = f.read()
    
    # Decode bytes to string
    try:
        text_content = content.decode('utf-8')
    except UnicodeDecodeError:
        text_content = content.decode('latin-1')
    
    # Track replacements
    replacement_status = {}
    modified_content = text_content
    
    # Process each replacement
    for old_pattern, new_pattern in replacements_dict.items():
        # Check if pattern exists
        if old_pattern in modified_content:
            # Count occurrences
            count = modified_content.count(old_pattern)
            # Replace all occurrences
            modified_content = modified_content.replace(old_pattern, new_pattern)
            replacement_status[old_pattern] = {
                'found': True,
                'replaced': count,
                'new_value': new_pattern
            }
            print(f"✅ Replaced '{old_pattern}' → '{new_pattern}' ({count} occurrence(s))")
        else:
            replacement_status[old_pattern] = {
                'found': False,
                'replaced': 0,
                'new_value': new_pattern
            }
            print(f"❌ Pattern '{old_pattern}' not found")
    
    # Determine output path
    if output_path is None:
        output_path = notebook_path
    
    # Write the modified content back to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"\n📁 File saved: {output_path}")
    return replacement_status
