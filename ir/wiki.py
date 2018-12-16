import wikipedia
import wikipedia.exceptions
from bs4 import BeautifulSoup
import os
import json
import unicodedata
from collections import defaultdict
from etl.table import Table


def get_attributes(html):
    attributes = defaultdict(dict)
    soup = BeautifulSoup(html, 'html.parser')
    infobox = soup.find('table', {'class': 'infobox'})
    if infobox:
        merged_top_row = None
        for tr in infobox.find_all('tr'):
            for sup in tr.find_all('sup'):
                sup.decompose()
            children = list(tr.children)

            if tr.get('class') == ['mergedtoprow']:
                if [child.name for child in children] == ['th']:
                    merged_top_row = children[0].get_text()
                else:
                    merged_top_row = None

            if [child.name for child in children] == ['th', 'td']:
                key = children[0].get_text()
                value = children[1].get_text()
                if tr.get('class') == ['mergedrow'] and merged_top_row:
                    attributes[merged_top_row][key] = value
                else:
                    attributes[key] = value

        return attributes
    else:
        return None


def get_wiki_info(query, title):
    try:
        page = wikipedia.page(title)
        html = page.html()
        return {'query': query, 'title': page.title, 'pageid': page.pageid, 'summary': page.summary,
                'attributes': get_attributes(html)}
    except wikipedia.exceptions.DisambiguationError as e:
        if title.lower() == e.options[0].lower():
            return get_wiki_info(query, e.options[1])
        else:
            return get_wiki_info(query, e.options[0])
    except wikipedia.exceptions.PageError:
        return {'query': query, 'title': None, 'pageid': None, 'summary': None, 'attributes': None}


if __name__ == '__main__':
    sample_dir = 'data/train_100_sample/0'
    for file in os.listdir(sample_dir):
        if file.endswith('.json') and file == '1438042988458.74_20150728002308-00063-ip-10-236-191-2_890299080_3.json':
            data = json.load(open(os.path.join(sample_dir, file)))
            if data['keyColumnIndex'] > 0:
                results = []
                key_column_index = data['keyColumnIndex']
                table = Table(data)
                label = table.get_header()[key_column_index]
                items = table.get_attributes()[key_column_index]
                for item in items:
                    print(item)
                    wiki_info = get_wiki_info(item, item)
                    print(wiki_info)
                    results.append(wiki_info)
                json.dump(results, open(os.path.join('ir', 'wiki', file), 'w+'), indent=4, ensure_ascii=False)
