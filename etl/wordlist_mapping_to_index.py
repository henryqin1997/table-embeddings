import json

input_wordlist = 'data/wordlist_v6_index.json'
input_wordlist_mapping = 'data/wordlist_v6_mapping.json'
output_wordlist_mapping_index = 'data/wordlist_v6_mapping_index.json'

if __name__ == '__main__':
    wordlist = json.load(open(input_wordlist))
    wordlist_mapping = json.load(open(input_wordlist_mapping))
    wordlist_mapping_index = list(range(len(wordlist) + 1))

    for k, v in wordlist_mapping.items():
        wordlist_mapping_index[wordlist[k]] = wordlist[v]

    json.dump(wordlist_mapping_index, open(output_wordlist_mapping_index, 'w+'), indent=4)
