import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp

from cytoolz import concat, curry, compose

from utils.io import count_data
import argparse


@curry
def process(data_dir, i):
    with open(join(data_dir, '{}.json'.format(i))) as f:
        data = json.loads(f.read())
    art_sents = data['article']
    abs_sents = data['abstract']
    art_sents_lower = [art_sent.lower() for art_sent in art_sents]
    abs_sents_lower = [abs_sent.lower() for abs_sent in abs_sents]

    new_data = {}
    new_data['article'] = art_sents_lower
    new_data['abstract'] = abs_sents_lower
    new_data['date'] = data['date']
    new_data['id'] = data['id']
    #new_data['extracted_m2o'] = data['extracted_m2o']
    new_data['topic'] = data['topic']
    new_data['hdl'] = data['hdl']
    new_data['general_descriptors'] = data['general_descriptors']
    new_data['descriptors'] = data['descriptors']

    with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
        json.dump(new_data, f, indent=4)


def label_mp(data, split):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(data, split)
    n_data = count_data(data_dir)
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(data_dir),
                                 list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def main(data, split):
    if split == 'all':
        for split in ['val', 'train', 'test']:
            label_mp(data, split)
    else:
        label_mp(data, split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Make extraction labels')
    )
    parser.add_argument('-data', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('-split', type=str, action='store', default='all',
                        help='The folder name that needs to produce candidates. all means process both train and val.')
    args = parser.parse_args()
    main(args.data, args.split)
