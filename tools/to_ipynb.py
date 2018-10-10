import os
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
import re
from redbaron import RedBaron
import yaml


with open('samples.yaml', 'r') as f:
    samples = yaml.load(f.read())


def should_concat(prev_type, cur_type):
    concat_types = ['comment', 'import', 'from_import', 'assignment']
    import_types = ['import', 'from_import']

    result = False

    if prev_type == cur_type and cur_type in concat_types:
        result = True

    if prev_type in import_types and cur_type in import_types:
        result = True

    return result

def py_to_ipynb(root, path, py_filename, git_clone=False, remove={}):
    py_filepath = os.path.join(root, path, py_filename)
    print('Converting {}'.format(py_filepath))

    ipynb_filename = py_filename.split('.')[0] + '.ipynb'
    ipynb_filepath = os.path.join(root, path, ipynb_filename)

    with open(py_filepath, 'r') as f:
        red = RedBaron(f.read())

    # detect whether the state has changed and thus need to flush the code up to that point before processing a node
    cells = []
    cell_source = [red[0].dumps()]
    prev_type = red[0].type

    for node in red[1:]:
        cur_type = node.type
        cur_code = node.dumps()

        # ignore blank lines
        if cur_type == 'endl':
            continue

        # ignore pylint comments
        if cur_type == 'comment' and 'pylint' in cur_code:
            continue

        # remove
        if cur_type in remove:
            remove_strs = remove[cur_type]

            for remove_str in remove_strs:
                cur_code = re.sub(remove_str, '', cur_code)

        if should_concat(prev_type, cur_type):
            cell_source.append(cur_code)

        else:
            content = '\n'.join(cell_source)
            if prev_type == 'comment':
                cell = new_markdown_cell(content)

            else:
                cell = new_code_cell(content)

            cells.append(cell)
            cell_source = [cur_code]

        prev_type = cur_type

    # last cell, expected to be the "if __name__" block
    cell_source = []

    # the value of the IfNode 
    node_value = node.value[0].value

    # just include all the lines
    for line in node_value:
        cell_source.append(line.dumps())

    content = '\n'.join(cell_source)
    cell = new_code_cell(content)
    cells.append(cell)

    # git clone
    if git_clone:
        with open('git_clone.p', 'r') as f:
            template = f.read()

        content = template.format(path=path)
        cell = new_code_cell(content)
        cells.insert(0, cell)

    notebook = new_notebook(cells=cells)

    # output
    with open(ipynb_filepath, 'w') as f:
        print("Writing: {}".format(ipynb_filepath))
        nbformat.write(notebook, f)


for sample, info in samples.iteritems():
    root = '..'
    path = info['path']
    filename = info['filename']
    git_clone = info['colab']['git-clone']
    remove = info['remove']

    py_to_ipynb(root, path, filename, git_clone=git_clone, remove=remove)

