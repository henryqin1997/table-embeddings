import json
import os
from .table import Table


def tables(files, prefix='webtables/', print_progress=False):
    if isinstance(files, str):
        files = json.load(open(files))

    for file in files:
        if print_progress:
            print(file)
        data = json.load(open(os.path.join(prefix, file), encoding='utf-8'))
        yield Table(data)
