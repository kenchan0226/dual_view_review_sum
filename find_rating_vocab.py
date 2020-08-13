import os
import json
from os.path import join
import re
import argparse
from collections import Counter
import pickle as pkl
from tqdm import tqdm
from nltk.corpus import stopwords
from string import punctuation


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
    #positive_word_list = read_word_list_from_file(positive_words_file)
    #negative_word_list = read_word_list_from_file(negative_words_file)
    #sentiment_word_list = positive_word_list + negative_word_list

    stop_words = set(stopwords.words('english'))

    out_dir = data_dir

    rating_1_vocab_counter = Counter()
    rating_2_vocab_counter = Counter()
    rating_3_vocab_counter = Counter()
    rating_4_vocab_counter = Counter()
    rating_5_vocab_counter = Counter()

    rating_1_vocab_counter_no_stop_word_and_punc = Counter()
    rating_2_vocab_counter_no_stop_word_and_punc = Counter()
    rating_3_vocab_counter_no_stop_word_and_punc = Counter()
    rating_4_vocab_counter_no_stop_word_and_punc = Counter()
    rating_5_vocab_counter_no_stop_word_and_punc = Counter()

    total_num_review = 0

    for i in tqdm(range(n_data)):
        total_num_review += 1
        js = json.load(open(join(split_dir, '{}.json'.format(i))))

        rating = js['overall']
        review_sent_list = js['reviewText']
        num_review_sents = len(review_sent_list)
        review_text = ' '.join(review_sent_list)
        review_word_list = review_text.split(' ')
        num_review_tokens = len(review_word_list)

        summary_sent_list = js['summary']
        summary_text = ' '.join(summary_sent_list)
        summary_word_list = summary_text.split(' ')
        num_summary_tokens = len(summary_word_list)
        summary_word_list_no_stop_word_and_punc = [word for word in summary_word_list if word not in stop_words and word not in punctuation]

        if rating == 1:
            target_vocab_counter = rating_1_vocab_counter
            target_vocab_counter_no_stop_word_and_punc = rating_1_vocab_counter_no_stop_word_and_punc
        elif rating == 2:
            target_vocab_counter = rating_2_vocab_counter
            target_vocab_counter_no_stop_word_and_punc = rating_2_vocab_counter_no_stop_word_and_punc
        elif rating == 3:
            target_vocab_counter = rating_3_vocab_counter
            target_vocab_counter_no_stop_word_and_punc = rating_3_vocab_counter_no_stop_word_and_punc
        elif rating == 4:
            target_vocab_counter = rating_4_vocab_counter
            target_vocab_counter_no_stop_word_and_punc = rating_4_vocab_counter_no_stop_word_and_punc
        elif rating == 5:
            target_vocab_counter = rating_5_vocab_counter
            target_vocab_counter_no_stop_word_and_punc = rating_5_vocab_counter_no_stop_word_and_punc
        else:
            raise ValueError
        target_vocab_counter.update(summary_word_list)
        target_vocab_counter_no_stop_word_and_punc.update(summary_word_list_no_stop_word_and_punc)


    with open(os.path.join(out_dir, "rating_1_vocab_counter.pkl"),
              'wb') as vocab_file:
        pkl.dump(rating_1_vocab_counter, vocab_file)

    with open(os.path.join(out_dir, "rating_2_vocab_counter.pkl"),
              'wb') as vocab_file:
        pkl.dump(rating_2_vocab_counter, vocab_file)

    with open(os.path.join(out_dir, "rating_3_vocab_counter.pkl"),
              'wb') as vocab_file:
        pkl.dump(rating_3_vocab_counter, vocab_file)

    with open(os.path.join(out_dir, "rating_4_vocab_counter.pkl"),
              'wb') as vocab_file:
        pkl.dump(rating_4_vocab_counter, vocab_file)

    with open(os.path.join(out_dir, "rating_5_vocab_counter.pkl"),
              'wb') as vocab_file:
        pkl.dump(rating_5_vocab_counter, vocab_file)

    with open(os.path.join(out_dir, "rating_1_vocab_counter_no_stop_word_and_punc.pkl"),
              'wb') as vocab_file:
        pkl.dump(rating_1_vocab_counter_no_stop_word_and_punc, vocab_file)

    with open(os.path.join(out_dir, "rating_2_vocab_counter_no_stop_word_and_punc.pkl"),
              'wb') as vocab_file:
        pkl.dump(rating_2_vocab_counter_no_stop_word_and_punc, vocab_file)

    with open(os.path.join(out_dir, "rating_3_vocab_counter_no_stop_word_and_punc.pkl"),
              'wb') as vocab_file:
        pkl.dump(rating_3_vocab_counter_no_stop_word_and_punc, vocab_file)

    with open(os.path.join(out_dir, "rating_4_vocab_counter_no_stop_word_and_punc.pkl"),
              'wb') as vocab_file:
        pkl.dump(rating_4_vocab_counter_no_stop_word_and_punc, vocab_file)

    with open(os.path.join(out_dir, "rating_5_vocab_counter_no_stop_word_and_punc.pkl"),
              'wb') as vocab_file:
        pkl.dump(rating_5_vocab_counter_no_stop_word_and_punc, vocab_file)



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
