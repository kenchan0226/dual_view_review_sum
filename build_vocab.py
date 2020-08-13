from os.path import join
import json
import os
import random
import argparse
from collections import Counter
import pickle as pkl
import re

def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def main(data_dir):
    split_dir = join(data_dir, "train")
    n_data = _count_data(split_dir)
    vocab_counter = Counter()
    for i in range(n_data):
        js = json.load(open(join(split_dir, '{}.json'.format(i))))

        summary = js['summary']
        summary_text = ' '.join(summary).strip()
        summary_word_list = summary_text.strip().split(' ')

        review = js['reviewText']
        review_text = ' '.join(review).strip()
        review_word_list = review_text.strip().split(' ')

        all_tokens = summary_word_list + review_word_list
        vocab_counter.update([t for t in all_tokens if t != ""])

    with open(os.path.join(data_dir, "vocab_cnt.pkl"),
              'wb') as vocab_file:
        pkl.dump(vocab_counter, vocab_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Preprocess review data')
    )
    parser.add_argument('-data_dir', type=str, action='store',
                        help='The directory of the data.')
    args = parser.parse_args()
    main(args.data_dir)
