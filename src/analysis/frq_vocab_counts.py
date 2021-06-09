'''
This script takes as input a json metadata file (as obtained by preprocessing agenda)
where we have entries 'title' and 'abstract'.
Then it filters by frequency discarding any types
appearing less than 5 times (like GraphWriter did).
Finally, it prints the number of items in the vocabulary.
'''

import argparse
import json
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument('json_file')
args = p.parse_args()

with open(args.json_file) as f:
    data = json.load(f)

frq = defaultdict(int)
for title in data['title']:
    for token in title:
        frq[token] += 1
for abstract in data['abstract']:
    for token in abstract:
        frq[token] += 1

# filtering
vocab = set()
for word in frq:
    if frq[word] >= 5:
        vocab.add(word)

# printing vocab size
print(len(vocab))
