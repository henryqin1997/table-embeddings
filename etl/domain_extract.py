import json
import os.path
from shutil import copyfile
from urllib.parse import urlparse

domain = 'http://www.usms.org/'
domain_filename = ''.join(x for x in domain if x.isalnum())

files = json.load(open('data/training_files.json'))
results = []

for file in files:
    data = json.load(open(os.path.join('data/train', file)))
    if 'url' in data:
        parsed_uri = urlparse(data['url'])
        result = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
        if result == domain:
            results.append(file)
            if not os.path.exists(os.path.join('domain_samples', domain_filename, os.path.dirname(file))):
                os.makedirs(os.path.join('domain_samples', domain_filename, os.path.dirname(file)))
            copyfile(os.path.join('webtables', file), os.path.join('domain_samples', domain_filename, file))
            print(os.path.join('domain_samples', domain_filename, file))

json.dump(results, open('data/domains/{}.json'.format(domain_filename), 'w+'), indent=4)
