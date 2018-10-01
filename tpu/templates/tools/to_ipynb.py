import os
import nbformat
from nbformat.v4 import new_code_cell,new_notebook
from redbaron import RedBaron

def should_concat(prev_type, cur_type):
    concat_types = ['comment', 'import', 'from_import', 'assignment']
    import_types = ['import', 'from_import']

    result = False

    if prev_type == cur_type and cur_type in concat_types:
        result = True

    if prev_type in import_types and cur_type in import_types:
        result = True

    return result

def py_to_ipynb(path, py_filename):
    print('Converting {}'.format(os.path.join(path, filename)))

    ipynb_filename = py_filename.split('.')[0] + '.ipynb'

    with open(os.path.join(path, py_filename), 'r') as f:
        red = RedBaron(f.read())

    # detect whether the state has changed and thus need to flush the code up to that point before processing a node
    sources = []
    cell_source = []

    prev_type = red[0].type
    # concat_state = [False]
    for node in red[1:]:
        cur_type = node.type

        # ignore blank lines
        if cur_type == 'endl':
            continue

        if should_concat(prev_type, cur_type):
            cell_source.append(node.dumps())
        else:
            sources.append('\n'.join(cell_source))
            cell_source = [node.dumps()]

        prev_type = cur_type

    # last cell, special handling
    cell_source = []

    # the value of the IfNode 
    node_value = node.value[0].value

    # just include all the argparse lines
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

        py_to_ipynb(path, filename)



