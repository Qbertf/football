import json
import sys
from pathlib import Path
from typing import List, Dict, Any

class NotebookToPythonGenerator:
    """Convert Jupyter notebook cells to Python dictionary"""
    
    def __init__(self, notebook_path: str):
        self.notebook_path = Path(notebook_path)
        self.cells = []
        
    def load_notebook(self) -> Dict[str, Any]:
        if not self.notebook_path.exists():
            raise FileNotFoundError(f"Notebook file not found: {self.notebook_path}")
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_cells(self) -> List[List[str]]:
        notebook = self.load_notebook()
        cells = notebook.get('cells', [])
        
        extracted_cells = []
        for cell in cells:
            source = cell.get('source', [])
            if isinstance(source, list):
                extracted_cells.append(source)
            elif isinstance(source, str):
                extracted_cells.append([source])
            else:
                extracted_cells.append([str(source)])
        
        self.cells = extracted_cells
        return extracted_cells
    
    def format_cell(self, content: List[str]) -> str:
        if not content:
            return '[]'
        if len(content) == 1:
            return f'[{repr(content[0])}]'
        lines = ',\n    '.join(repr(line) for line in content)
        return f'[\n    {lines}\n]'
    
    def generate_python_code(self) -> str:
        if not self.cells:
            self.extract_cells()
        
        notebook_name = self.notebook_path.name
        total_cells = len(self.cells)
        notebook_stem = self.notebook_path.stem
        
        lines = [
            '#!/usr/bin/env python3',
            '# -*- coding: utf-8 -*-',
            '"""',
            f'Generated from notebook: {notebook_name}',
            f'Total cells: {total_cells}',
            'All cells are stored in a dictionary called "cells"',
            '"""',
            '',
            'import json',
            'import sys',
            'from pathlib import Path',
            '',
            '# ============================================',
            '# CELLS DICTIONARY',
            '# ============================================',
            '',
            '# Initialize empty dictionary',
            'cells = {}',
            '',
            ''
        ]
        
        # Add each cell with cells.update()
        for idx, content in enumerate(self.cells, start=1):
            formatted = self.format_cell(content)
            lines.append(f'# Cell_{idx:04d}')
            lines.append(f'cells.update({{{idx}: {formatted}}})')
            lines.append('')
        
        # Add utility functions
        lines.extend([
            '',
            '# ============================================',
            '# UTILITY FUNCTIONS',
            '# ============================================',
            '',
            'def create_notebook(cell_list, output_path="new_notebook.ipynb"):',
            '    """Create a Jupyter notebook from cell contents"""',
            '    notebook = {',
            '        "cells": [],',
            '        "metadata": {',
            '            "kernelspec": {',
            '                "display_name": "Python 3",',
            '                "language": "python",',
            '                "name": "python3"',
            '            },',
            '            "language_info": {',
            '                "codemirror_mode": {"name": "ipython", "version": 3},',
            '                "file_extension": ".py",',
            '                "mimetype": "text/x-python",',
            '                "name": "python",',
            '                "nbconvert_exporter": "python",',
            '                "pygments_lexer": "ipython3",',
            '                "version": "3.8.0"',
            '            }',
            '        },',
            '        "nbformat": 4,',
            '        "nbformat_minor": 4',
            '    }',
            '    ',
            '    for cell_content in cell_list:',
            '        if isinstance(cell_content, str):',
            '            cell_content = [cell_content]',
            '        elif not isinstance(cell_content, list):',
            '            cell_content = [str(cell_content)]',
            '        ',
            '        # Detect cell type',
            '        cell_type = "code"',
            '        if cell_content and isinstance(cell_content[0], str):',
            '            first_line = cell_content[0].strip()',
            '            if first_line.startswith("#") and not first_line.startswith("#!"):',
            '                cell_type = "markdown"',
            '        ',
            '        notebook["cells"].append({',
            '            "cell_type": cell_type,',
            '            "metadata": {},',
            '            "source": cell_content,',
            '            "outputs": [] if cell_type == "code" else None,',
            '            "execution_count": None if cell_type == "code" else None',
            '        })',
            '    ',
            '    output_path = Path(output_path)',
            '    if not output_path.suffix:',
            '        output_path = output_path.with_suffix(".ipynb")',
            '    ',
            '    with open(output_path, "w", encoding="utf-8") as f:',
            '        json.dump(notebook, f, ensure_ascii=False, indent=2)',
            '    ',
            '    print(f"✅ Notebook created: {output_path}")',
            '    print(f"📊 Total cells: {len(cell_list)}")',
            '    return output_path',
            '',
            '',
            'def get_cell(index):',
            '    """Get a specific cell by index (1-based)"""',
            '    if index in cells:',
            '        return cells[index]',
            '    raise KeyError(f"Cell {index} not found (1 to {len(cells)})")',
            '',
            '',
            'def get_cells(start=1, end=None):',
            '    """Get a range of cells (1-based)"""',
            '    if end is None:',
            '        end = len(cells)',
            '    return [cells[i] for i in range(start, end + 1)]',
            '',
            '',
            'def get_all_cells():',
            '    """Get all cell contents as a list of lists"""',
            '    return [cells[i] for i in sorted(cells.keys())]',
            '',
            '',
            'def show_cell_info():',
            f'    """Display information about extracted cells from {notebook_name}"""',
            '    print("=" * 60)',
            f'    print(f"📓 Notebook: {notebook_name}")',
            f'    print(f"📊 Total cells: {total_cells}")',
            '    print("=" * 60)',
            '    ',
            '    for idx in sorted(cells.keys()):',
            '        content = cells[idx]',
            '        lines = len(content) if isinstance(content, list) else 0',
            '        preview = str(content[0])[:40] if content and len(content) > 0 else "Empty"',
            '        print(f"  Cell_{idx:04d} | Lines: {lines:3d} | Preview: {preview}...")',
            '',
            '',
            '# ============================================',
            '# MAIN EXECUTION',
            '# ============================================',
            '',
            'if __name__ == "__main__":',
            '    if len(sys.argv) > 1:',
            '        # Create new notebook from all cells',
            '        output_file = sys.argv[1]',
            '        print("=" * 60)',
            '        print("📝 CREATING NEW NOTEBOOK")',
            '        print("=" * 60)',
            '        ',
            '        cell_list = get_all_cells()',
            '        if cell_list:',
            '            print(f"📊 Using {len(cell_list)} cells")',
            '            create_notebook(cell_list, output_file)',
            '        else:',
            '            print("❌ No cells found!")',
            '    else:',
            '        # Show info',
            '        print("=" * 60)',
            '        print("📓 EXTRACTED CELLS")',
            '        print("=" * 60)',
            '        show_cell_info()',
            '        ',
            '        print("\\n" + "=" * 60)',
            '        print("💡 USAGE:")',
            '        print("=" * 60)',
            f'        print(f"  python {notebook_stem}.py newnotebook.ipynb")',
            '        print("  ")',
            '        print("  # Access individual cells:")',
            '        print("  from oxford import cells")',
            '        print("  first_cell = cells[1]           # First cell")',
            '        print("  second_cell = cells[2]          # Second cell")',
            '        print("  ")',
            '        print("  # Create notebook from specific cells:")',
            '        print("  from oxford import cells, create_notebook")',
            '        print("  create_notebook([cells[1], cells[3]], \\"selected.ipynb\\")")',
            '        print("  ")',
            '        print("  # Get range of cells:")',
            '        print("  from oxford import get_cells")',
            '        print("  selected = get_cells(1, 10)  # Cells 1 to 10")',
            '        print("  create_notebook(selected, \\"first_10_cells.ipynb\\")")',
            ''
        ])
        
        return '\n'.join(lines)
    
    def save_to_file(self, output_path: str = None):
        if output_path is None:
            output_path = self.notebook_path.with_suffix('.py')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_python_code())
        
        print(f"✅ Generated: {output_path}")
        print(f"📊 Extracted {len(self.cells)} cells")
        return output_path


def main():
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
    else:
        notebook_path = "sample_notebook.ipynb"
    
    try:
        print("=" * 60)
        print("🔄 CONVERTING NOTEBOOK TO PYTHON")
        print("=" * 60)
        
        generator = NotebookToPythonGenerator(notebook_path)
        cells = generator.extract_cells()
        print(f"📖 Found {len(cells)} cells")
        
        output_file = generator.save_to_file()
        
        print("\n" + "=" * 60)
        print("✅ DONE!")
        print("=" * 60)
        print(f"📁 Output: {output_file}")
        print(f"\n💡 Run: python {Path(output_file).stem}.py newnotebook.ipynb")
        print("\n📌 Using dictionary format: cells[1], cells[2], ...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Usage: python notebook_to_python.py oxford.ipynb")


if __name__ == "__main__":
    main()