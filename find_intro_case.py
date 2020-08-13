import os
import json
from os.path import join
import re
import argparse


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


def main(data_dir, split, positive_words_file, negative_words_file):
    split_dir = join(data_dir, split)
    n_data = _count_data(split_dir)
    positive_word_list = read_word_list_from_file(positive_words_file)
    negative_word_list = read_word_list_from_file(negative_words_file)
    sentiment_word_list = positive_word_list + negative_word_list

    total_num_review = 0

    for i in range(n_data):
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

        num_matched_sent_words = 0

        for w in summary_word_list:
            if w in sentiment_word_list:
                num_matched_sent_words += 1

        if num_matched_sent_words > 0 and num_summary_tokens > 5:
            print("{}.json".format(i))
            print("Rating: {}".format(rating))
            print("Summary: ")
            print(summary_text)
            print("Review: ")
            print(review_text)
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Preprocess review data')
    )
    parser.add_argument('-data_dir', type=str, action='store',
                        help='The directory of the data.')
    parser.add_argument('-split', type=str, action='store',
                        help='train or val or test.')
    parser.add_argument('-positive_words_file', type=str, action='store',
                        help='Path the file of positive sentiment words.')
    parser.add_argument('-negative_words_file', type=str, action='store',
                        help='Path the file of negative sentiment words.')

    args = parser.parse_args()

    main(args.data_dir, args.split, args.positive_words_file, args.negative_words_file)
