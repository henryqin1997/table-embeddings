import json
import os.path
from shutil import copyfile
from collections import defaultdict
from operator import itemgetter
from urllib.parse import urlparse

files = json.load(open('data/training_files.json'))
url_dict = defaultdict(int)

for file in files:
    data = json.load(open(os.path.join('data/train', file)))
    if 'url' in data:
        parsed_uri = urlparse(data['url'])
        result = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)

        url_dict[result] += 1

url_dict = sorted(url_dict.items(), key=itemgetter(1), reverse=True)

json.dump(url_dict, open('data/domain_dict.json','w+'),indent=4)
