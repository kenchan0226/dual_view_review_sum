""" make reference text files needed for ROUGE evaluation """
""" Adapted from https://github.com/ChenRocks/fast_abs_rl """

import json
import os
from os.path import join, exists
from time import time
from datetime import timedelta
from utils import io
import argparse
import numpy as np
import pickle as pkl


def dump(split, data_dir):
    start = time()
    print('start processing {} split...'.format(split))
    split_dir = join(data_dir, split)
    dump_dir = join(data_dir, 'refs', split)
    rating_dir = join(data_dir, 'ratings', split)
    n_data = io.count_data(split_dir)
    all_rating_list = []
    for i in range(n_data):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100*i/n_data),
              end='')
        with open(join(split_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        abs_sents = data['summary']
        abs_sents = [sent.lower() for sent in abs_sents]
        with open(join(dump_dir, '{}.ref'.format(i)), 'w') as f:
            f.write(io.make_html_safe('\n'.join(abs_sents)))
        all_rating_list.append(int(data['overall'])-1)
    # write rating file
    #print(all_rating_list)
    all_rating_array = np.array(all_rating_list)
    if not exists(rating_dir):
        os.makedirs(rating_dir)
    with open(join(rating_dir, 'gold_ratings.pkl'), 'wb') as f:
        pkl.dump(all_rating_array, f, pkl.HIGHEST_PROTOCOL)
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def main(split, data_dir):
    if split == 'all':
        for split in ['val', 'test']:  # evaluation of train data takes too long
            if not exists(join(data_dir, 'refs', split)):
                os.makedirs(join(data_dir, 'refs', split))
            dump(split, data_dir)
    else:
        if not exists(join(data_dir, 'refs', split)):
            os.makedirs(join(data_dir, 'refs', split))
        dump(split, data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Make evaluation reference.'))
    parser.add_argument('-data', type=str, action='store', default='',
                        help='The path of the data directory.')
    parser.add_argument('-split', type=str, action='store', default='all',
                        help='The folder name that needs to produce reference. all means process both val and test.')
    args = parser.parse_args()

    main(args.split, args.data)

