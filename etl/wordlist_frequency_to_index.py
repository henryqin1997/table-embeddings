import json

input_wordlist = 'data/wordlist_v4.json'
output_wordlist = 'data/wordlist_v4_index.json'

if __name__ == '__main__':
    wordlist_frequency = json.load(open(input_wordlist))
    wordlist_index = dict([(x[0], i) for i, x in enumerate(wordlist_frequency.items())])
    json.dump(wordlist_index, open(output_wordlist, 'w+'), indent=4)
