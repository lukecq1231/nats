#!/usr/bin/python
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('origin', type=str)
parser.add_argument('new', type=str)
args = parser.parse_args()

corpus = args.input
summary = args.origin
new_summary = args.new

# flag
extractive = 0
remove_eos = 1 

all_words = []
with open(corpus, 'r') as f:
    for line in f:
        words = line.strip().split()
        all_words.append(words)

with open(new_summary, 'w') as fo:
    with open(summary, 'r') as f:
        for line, words in zip(f, all_words):
            words_and_poss = line.strip().split()
            y = words_and_poss[::2]
            pos = words_and_poss[1::2]
            pos = map(lambda x: re.sub(r'\[|\]', "", x), pos)
            pos = map(int, pos)
            # print y, pos
            for a, b in zip(y, pos):
                if remove_eos and a == '<EOS>':
                    continue
                if not extractive:
                    # only replace unk
                    if a == 'UNK' and b < len(words):
                        if words[b] == '<EOS>':
                            continue
                        print >>fo, words[b],
                    else:
                        print >>fo, a,
                else:
                    # replace all
                    print >>fo, a,
            print >>fo
