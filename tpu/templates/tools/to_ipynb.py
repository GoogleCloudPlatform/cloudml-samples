import os
import nbformat
from nbformat.v4 import new_code_cell,new_notebook
from redbaron import RedBaron

def py_to_ipynb(path, py_filename):
    ipynb_filename = py_filename.split('.')[0] + '.ipynb'

    with open(os.path.join(path, py_filename), 'r') as f:
        red = RedBaron(f.read())

    # types of nodes to be put into the same cell
    in_comment = False
    in_import = False

    sources = []
    cell_source = []
    for node in red:
        if node.type == 'endl':
            continue

        elif node.type == 'comment':
            if not in_comment:
                # flush
                sources.append('\n'.join(cell_source))
                cell_source = []

            cell_source.append(node.dumps())
            in_comment = True
            in_import = False

        elif node.type in ['import', 'from_import']:
            if not in_import:
                # flush
                sources.append('\n'.join(cell_source))
                cell_source = []

            cell_source.append(node.dumps())
            in_comment = False
            in_import = True

        else:
            # flush
            sources.append('\n'.join(cell_source))
            cell_source = []

            cell_source.append(node.dumps())
            in_comment = False
            in_import = False

    # last cell, special handling
    cell_source = []

    # the value of the IfNode 
    node_value = node.value[0].value
    for line in node_value:
        cell_source.append(line.dumps())

    sources.append('\n'.join(cell_source))

    # build cells and notebook
    cells = [new_code_cell(source=source) for source in sources if source]
    notebook = new_notebook(cells=cells)

    # output
    with open(os.path.join(path, ipynb_filename), 'w') as f:
        nbformat.write(notebook, f)


root = '..'
for path, dirs, filenames in os.walk(root):
    if 'tpu' not in path:
        continue

    for filename in filenames:
        if filename.startswith('__') or not filename.endswith('.py'):
            continue

        print('Converting {}'.format(os.path.join(path, filename)))
        py_to_ipynb(path, filename)



