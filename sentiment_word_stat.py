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

    total_num_summ_with_sent_word = 0
    total_num_sent_words_in_summ = 0
    total_num_review = 0
    total_num_review_sents = 0
    total_num_review_tokens = 0
    total_num_summary_tokens = 0
    max_review_tokens = 0
    max_summary_tokens = 0
    total_num_present_summary_tokens = 0
    total_num_short_review = 0
    total_num_short_summary = 0

    for i in range(n_data):
        total_num_review += 1
        js = json.load(open(join(split_dir, '{}.json'.format(i))))

        review_sent_list = js['reviewText']
        num_review_sents = len(review_sent_list)
        review_text = ' '.join(review_sent_list)
        review_word_list = review_text.split(' ')
        num_review_tokens = len(review_word_list)

        summary_sent_list = js['summary']
        summary_text = ' '.join(summary_sent_list)
        summary_word_list = summary_text.split(' ')
        num_summary_tokens = len(summary_word_list)
        if num_summary_tokens < 2:
            total_num_short_summary += 1
        if num_review_tokens < 8:
            total_num_short_review += 1
        num_matched_sent_words = 0
        for w in summary_word_list:
            if w in sentiment_word_list:
                num_matched_sent_words += 1
            if w in review_word_list:
                total_num_present_summary_tokens += 1

        if num_matched_sent_words > 0:
            total_num_summ_with_sent_word += 1
            total_num_sent_words_in_summ += num_matched_sent_words

        total_num_review_sents += num_review_sents
        total_num_review_tokens += num_review_tokens
        total_num_summary_tokens += num_summary_tokens

        if num_review_tokens > max_review_tokens:
            max_review_tokens = num_review_tokens
        if num_summary_tokens > max_summary_tokens:
            max_summary_tokens = num_summary_tokens

    print("% of summary contains sentiment words:\t{:.2f}".format(total_num_summ_with_sent_word/total_num_review * 100))
    print("avg # of sentiment word per summary:\t{:.3f}".format(total_num_sent_words_in_summ/total_num_review))
    print("avg # tokens in summary:\t{:.3f}".format(total_num_summary_tokens/total_num_review))
    print("avg # tokens in review:\t{:.3f}".format(total_num_review_tokens/total_num_review))
    print("avg # sentences in review:\t{:.3f}".format(total_num_review_sents/total_num_review))
    print("max # tokens in summary:\t{}".format(max_summary_tokens))
    print("max # tokens in review:\t{}".format(max_review_tokens))
    print("% of present tokens in summary:\t{:.2f}".format(total_num_present_summary_tokens/total_num_summary_tokens * 100))

    print("# short reviews:\t{}".format(total_num_short_review))
    print("# short summaries:\t{}".format(total_num_short_summary))

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
