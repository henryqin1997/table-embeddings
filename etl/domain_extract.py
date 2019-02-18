import json
import os.path
from collections import defaultdict
from urllib.parse import urlparse

domain = 'http://www.usms.org/'

files = json.load(open('data/training_files.json'))
results = []

for file in files:
    data = json.load(open(os.path.join('data/train', file)))
    if 'url' in data:
        parsed_uri = urlparse(data['url'])
        result = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
        if result == domain:
            results.append(file)

json.dump(results, open('data/domains/{}.json'.format(''.join(x for x in domain if x.isalnum())), 'w+'), indent=4)
