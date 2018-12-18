import json
import os.path
from shutil import copyfile
from collections import defaultdict
from operator import itemgetter
from urllib.parse import urlparse
from etl.table import Table

files = json.load(open('data/training_files.json'))
url_dict = defaultdict(int)

for file in files:
    data = json.load(open(os.path.join('data/train', file)))
    table = Table(data)
    if len(table.get_attributes()) < 5 or len(table.get_entities()) < 5:
        continue
    if 'url' in data:
        parsed_uri = urlparse(data['url'])
        result = '{uri.scheme}://{uri.netloc}/ {schema}'.format(uri=parsed_uri,
                                                                schema=[label.lower() for label in table.get_header()])

        url_dict[result] += 1

url_dict = sorted(url_dict.items(), key=itemgetter(1), reverse=True)

json.dump(url_dict, open('data/domain_schema_dict.json', 'w+'), indent=4)
