import os
import json
from os.path import join, exists
import re
import argparse
import numpy as np
import pickle as pkl


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def read_word_list_from_file(filename):
    with open(filename) as f:
        word_list = [l.strip() for l in f.readlines()]
    return word_list


def main(data_dir, split):
    split_dir = join(data_dir, split)
    n_data = _count_data(split_dir)
    all_ratings = np.zeros(n_data)
    rating_count = np.zeros(5)
    rating_dir = join(data_dir, 'ratings', split)
    if not exists(rating_dir):
        os.makedirs(rating_dir)

    #zero_flag = True

    for i in range(n_data):
        js = json.load(open(join(split_dir, '{}.json'.format(i))))
        rating = int(js['overall'])
        all_ratings[i] = rating
        rating_count[rating-1] += 1

        #if zero_flag and rating == 1:
        #    print(i)
        #    zero_flag = False

    print("Average rating: {}".format(all_ratings.mean()))
    print("Rating count:")
    print(rating_count)
    print("Rating ratio:")
    normalized_rating_count = rating_count/rating_count.sum()
    print(normalized_rating_count)
    print("Class weights")
    print(1.0/normalized_rating_count)

    all_ratings = all_ratings -1

    with open(join(rating_dir, 'gold_ratings.pkl'), 'wb') as f:
        pkl.dump(all_ratings, f, pkl.HIGHEST_PROTOCOL)

    with open(join(rating_dir, 'rating_count.pkl'), 'wb') as f:
        pkl.dump(rating_count, f, pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Preprocess review data')
    )
    parser.add_argument('-data_dir', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('-split', type=str, action='store',
                        help='train or val or test.')

    args = parser.parse_args()

    main(args.data_dir, args.split)
