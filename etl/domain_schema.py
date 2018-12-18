import json
import os.path
from shutil import copyfile
from collections import defaultdict
from operator import itemgetter
from urllib.parse import urlparse
from etl.table import Table

files = json.load(open('data/training_files.json'))
domain_schema_dict = defaultdict(int)
domain_schema_files_dict = {}

for file in files:
    data = json.load(open(os.path.join('data/train', file)))
    table = Table(data)
    if len(table.get_attributes()) < 5 or len(table.get_entities()) < 5:
        continue
    if 'url' in data:
        parsed_uri = urlparse(data['url'])
        result = '{uri.scheme}://{uri.netloc}/ {schema}'.format(uri=parsed_uri,
                                                                schema=[label.lower() for label in table.get_header()])

        domain_schema_dict[result] += 1
        if domain_schema_dict[result] == 1:
            domain_schema_files_dict[result] = file

domain_schema_dict = dict(sorted(domain_schema_dict.items(), key=itemgetter(1), reverse=True))
domain_schema_files_dict = dict(sorted(domain_schema_files_dict.items(), key=lambda item: domain_schema_dict[item[0]],
                                  reverse=True))

json.dump(domain_schema_dict, open('data/domain_schema_dict.json', 'w+'), indent=4)
json.dump(domain_schema_files_dict, open('data/domain_schema_files_dict.json', 'w+'), indent=4)
