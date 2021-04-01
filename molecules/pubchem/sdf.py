# Copyright 2019 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# This contains a simple parser for SDF files.

from __future__ import absolute_import

import json

MOLECULE_START = "-OEChem-"
MOLECULE_END = "$$$$"
MDF_SECTION = "MDF"
SECTION_PREFIX = "<PUBCHEM_"


def parse_molecules(raw_lines):
    """Generator that yields raw molecules."""
    molecule = None
    section = None
    for raw_line in raw_lines:
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8")
        line = raw_line.lstrip(">").strip()
        if not line:
            continue

        # If we find a new molecule section, yield the last molecule we were
        # parsing and initialize a new one
        if MOLECULE_START in line:
            molecule = {}
            section = molecule[MDF_SECTION] = []
            continue

        if molecule is None:
            continue

        # This is the start of a new section
        if line.startswith(SECTION_PREFIX):
            section = molecule[line] = []

        # This delimits the end of a molecule
        elif line == MOLECULE_END:
            section = None
            yield json.dumps(molecule)
            molecule = None

        # It didn't match anything else, so it must be content for a section
        elif section is not None:
            section.append(raw_line)

    # If there's a last unprocessed molecule, yield it
    if molecule is not None:
        yield json.dumps(molecule)
