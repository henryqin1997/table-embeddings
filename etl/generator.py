import json
import os
from .table import Table


def tables(files, prefix='webtables/'):
    if isinstance(files, str):
        files = json.load(open(files))

    for file in files:
        data = json.load(open(os.path.join(prefix, file), encoding='utf-8'))
        yield Table(data)
