# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import astor
import os
import re
import nbformat
from nbformat.v4 import new_code_cell
from nbformat.v4 import new_markdown_cell
from nbformat.v4 import new_notebook
import yaml


STYLES = {
    'colab': {
        # code template to be added right after the license.
        'precells': ['templates/colab_pre.p'],
        # TPU specific code template to be added right before running main.
        'tpu_precells': ['templates/colab_tpu.p'],
        # TPU specific code template to be added at the end.
        'tpu_postcells': None,
        # additional path segment.
        'prefix': ''
    },
    'notebooks': {
        'precells': ['templates/notebooks_pre.p'],
        'tpu_precells': [
            'templates/notebooks_tpu_args.p',
            'templates/notebooks_tpu_create.p'
        ],
        'tpu_postcells': ['templates/notebooks_tpu_post.p'],
        'prefix': 'notebooks'
    }
}


# Only samples registered and configured in samples.yaml will be converted.
with open('samples.yaml', 'r') as f:
    samples = yaml.load(f.read())


def should_concat(prev_type, cur_type):
    """This function contains the logic deciding if the current node should be grouped with the previous node in the same notebook cell.

    Args:
    prev_type: (str) type of the previous node.
    cur_type: (str) type of the current node.

    Returns
    A Boolean
    """
    concat_types = ['Import', 'ImportFrom', 'Assign']
    import_types = ['Import', 'ImportFrom']

    if prev_type == cur_type and cur_type in concat_types:
        return True

    if prev_type in import_types and cur_type in import_types:
        return True

    return False


class BoundaryVisitor(ast.NodeVisitor):
    def __init__(self):
        self.boundary = 0

    def generic_visit(self, node):
        lineno = getattr(node, 'lineno', 0)
        self.boundary = max(self.boundary, lineno)

        ast.NodeVisitor.generic_visit(self, node)


def get_boundary(node):
    """Get the boundaries of code representing node."""
    # lineno starts from 1
    lineno = getattr(node, 'lineno', 0)
    top = lineno - 1

    bv = BoundaryVisitor()
    bv.visit(node)
    bottom = bv.boundary

    return top, bottom


def process_between(group):
    """Process lines between nodes, that is, comments."""

    # Keep only comments
    group = [line for line in group if line.strip().startswith('#')]
    group = [re.sub(r'^#', '', line) for line in group]
    # Markdown could interpret this as header.
    group = [re.sub(r'={3,}', '', line) for line in group]

    return group


def process_node(group, cur_type, remove=None):
    """Remove portions of the code based on the config."""
    if remove is None or cur_type not in remove:
        return group

    result = []
    remove_strs = remove[cur_type]

    for line in group:
        for remove_str in remove_strs:
            line = re.sub(remove_str, '', line).strip()
            if line:
                result.append(line)

    return result


def code_cell(group, remove=None):
    source = '\n'.join(group).strip()
    return new_code_cell(source)


def markdown_cell(group):
    # Two spaces for markdown line break
    source = '  \n'.join(group).strip()
    return new_markdown_cell(source)


def add_cell(cells, filename, insert=None, **kwargs):
    """Modifies cells in place by adding the content of filename as a new cell in position `insert`.
    When `insert` is None, the new cell is appended to cells.
    Allows optional formatting arguments as kwargs."""
    with open(filename, 'r') as f:
        code_source = f.read()

    if kwargs:
        code_source = code_source.format(**kwargs)

    new_cell = new_code_cell(code_source)

    if insert is None:
        cells.append(new_cell)
    else:
        cells.insert(insert, new_cell)

    return cells


def py_to_ipynb(root, path, py_filename, style, remove=None):
    """This function converts the .py file at <root>/<path>/<py_filename> into a .ipynb of the same name in <root>/<path>.

    - Consecutive comments are grouped into the same cell.
    - Comments are turned into markdown cells.
    - The last part of the .py file is expected to be an `if __name__ == '__main__':` block.

    Args:
    style: 'colab' or 'notebooks'.
    remove: (None or dict) A Dict describing what code to be removed according to specified node type.

    Returns
    None
    """
    style_dict = STYLES[style]
    prefix = style_dict['prefix']

    py_filepath = os.path.join(root, path, py_filename)
    print('Converting {}'.format(py_filepath))

    ipynb_filename = py_filename.split('.')[0] + '.ipynb'
    ipynb_filepath = os.path.join(root, prefix, path, ipynb_filename)

    with open(py_filepath, 'r') as py_file:
        source = py_file.read()

    module = ast.parse(source, filename=py_filepath)
    lines = source.split('\n')

    cells = []
    cell_source = []            
    prev_type = None
    start = 0

    # main processing loop
    for node in module.body:
        cur_type = type(node).__name__
        top, bottom = get_boundary(node)

        # special handling for dangling lines
        if cur_type in ['Import', 'ImportFrom', 'Expr']:
            # print(astor.to_source(node))
            code_lines = astor.to_source(node).strip().split('\n')
            cur_group = process_node(code_lines, cur_type, remove)
        else:
            code_lines = lines[top:bottom]
            cur_group = process_node(code_lines, cur_type, remove)

        # group of lines between ast nodes
        between = process_between(lines[start:top])
        
        if between:
            # flush cell_source
            if cell_source:
                cells.append(code_cell(cell_source))

            cells.append(markdown_cell(between))

            # get current node source, check later if need to concatenate
            cell_source = cur_group

        else: # no between lines, check if need to concatenate
            # handle first node
            if prev_type is None:
                # prev_type = cur_type
                pass
            elif should_concat(prev_type, cur_type):
                cell_source.extend(cur_group)

            else: # flush
                cells.append(code_cell(cell_source))
                cell_source = cur_group

        prev_type = cur_type
        start = bottom

    # handle last cell
    # Skipping the line `if __name__ == '__main__':` and removing indentation
    indent = node.body[0].col_offset
    cell_source = [line[indent:] for line in cell_source][1:]

    # Add precell
    if style_dict['precells'] is not None:
        # we might be inserting multiple cells at the same position, so reversing the ordering to keep the correct order.
        for filename in reversed(style_dict['precells']):
            add_cell(cells, filename, insert=1, path=path)

    # special handling for tpu samples.
    if 'tpu' in py_filepath and style_dict['tpu_precells'] is not None:
        cs0 = [line for line in cell_source if 'main(args)' not in line]
        cells.append(code_cell(cs0))

        for filename in style_dict['tpu_precells']:
            add_cell(cells, filename)

        cs2 = 'main(args)'
        cells.append(new_code_cell(cs2))

    else:
        cells.append(code_cell(cell_source))

    # Add tpu postcell
    if 'tpu' in py_filepath and style_dict['tpu_postcells'] is not None:
        for filename in style_dict['tpu_postcells']:
            add_cell(cells, filename)

    notebook = new_notebook(cells=cells)

    # output
    outpath, _ = os.path.split(ipynb_filepath)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    with open(ipynb_filepath, 'w') as ipynb_file:
        nbformat.write(notebook, ipynb_file)

if __name__ == '__main__':
    for sample, info in samples.iteritems():
        root = '..'
        path = info['path']
        filename = info['filename']
        remove = info.get('remove', None)

        for style in ['colab', 'notebooks']:
            py_to_ipynb(root, path, filename, style, remove=remove)


    # Testing, make sure sample.py has the same output.
    with open('sample.ipynb', 'r') as f:
        sample_ipynb = f.read()

    with open('frozen_sample.ipynb', 'r') as f:
        frozen_sample_ipynb = f.read()

    assert sample_ipynb == frozen_sample_ipynb, "The sample.ipynb is different."
