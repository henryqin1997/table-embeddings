import json


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    print(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)


file1 = 'data/wordlist_v2.json'
file2 = 'hadoop/output2/part-00000'

list1 = list(json.load(open(file1)).keys())[:1000]
list2 = list(map(lambda line: line[line.index('\t') + 1:-1], open(file2).readlines()[:1000]))
print(jaccard_similarity(list1, list2))

print(set(list1) - set(list1).intersection(list2))
print(set(list2) - set(list1).intersection(list2))