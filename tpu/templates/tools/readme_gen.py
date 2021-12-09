import glob
import os
import re
import yaml

root = '..'

def colab_url(filename):
    return 'https://colab.research.google.com/github/GoogleCloudPlatform/cloudml-samples/blob/main/tpu/templates/{}'.format(filename)

def format(path):
    if not path:
        return ''

    formatted = ''

    py_filenames = glob.glob(os.path.join(root, path, '*.py'))
    py_filenames = [fn[3:] for fn in py_filenames if '__' not in fn]

    ipynb_filenames = glob.glob(os.path.join(root, path, '*.ipynb'))
    ipynb_filenames = [fn[3:] for fn in ipynb_filenames]

    for py_filename in py_filenames:
        formatted += '[{}]({})<br>'.format(py_filename.split('/')[-1], py_filename)

    for ipynb_filename in ipynb_filenames:
        formatted += '[{}]({}) [[Colab]]({})<br>'.format(ipynb_filename.split('/')[-1], ipynb_filename, colab_url(ipynb_filename))

    return formatted


with open('samples.yaml', 'r') as f:
    samples = yaml.load(f.read())

rows = []
columns = []

for k, d in samples.items():
    rows.append(k)
    columns.extend(d.keys())

columns = sorted(list(set(columns)))
rows = sorted(list(set(rows)))

table_str = '\n ... | '

table_str += ' | '.join(columns)
table_str += '\n --- | '
table_str += ' | '.join(['---']*len(columns))

for row in rows:
    table_str += '\n {} | '.format(row)
    table_str += ' | '.join([format(samples[row].get(column, '')) for column in columns])

with open('README_BASE.md', 'r') as f:
    readme_base = f.read()

readme = re.sub('<TABLE>', table_str, readme_base)

with open('../README.md', 'w') as f:
    f.write(readme)

