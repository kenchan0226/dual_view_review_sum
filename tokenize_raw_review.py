from os.path import join
import json
from nltk.parse import CoreNLPParser
import os
import random
import argparse
from collections import Counter
import pickle as pkl
import nltk

corenlp_parser = CoreNLPParser(url='http://localhost:9000')


def main(raw_data_file):
    out_filename = os.path.splitext(raw_data_file)[0] + '_tokenized.json'

    with open(raw_data_file) as f_in:
        print("Reading {}".format(raw_data_file))
        all_lines = f_in.readlines()
        print("Finish Reading")
    num_processed_review = 0

    with open(out_filename, 'w') as f_out:
        for review_i, line in enumerate(all_lines):
            js = json.loads(line.strip())
            summary = js['summary']
            summary_word_list = list(corenlp_parser.tokenize(summary.strip()))
            summary_word_list = [w.lower() for w in summary_word_list]
            summary_tokenized = ' '.join(summary_word_list)
            summary_tokenized_sent_list = nltk.sent_tokenize(summary_tokenized)
            review = js['reviewText']
            review_word_list = list(corenlp_parser.tokenize(review.strip()))
            review_word_list = [w.lower() for w in review_word_list]
            review_tokenized = ' '.join(review_word_list)
            js['summary'] = summary_tokenized_sent_list
            review_tokenized_sent_list = nltk.sent_tokenize(review_tokenized)
            js['reviewText'] = review_tokenized_sent_list

            # write one line to output file
            f_out.write(json.dumps(js) + '\n')
            num_processed_review += 1
            if num_processed_review % 10000 == 0:
                print("Processed {} samples".format(num_processed_review))

    print("{} processed samples".format(num_processed_review))
    print("Finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Preprocess review data')
    )
    parser.add_argument('-raw_data_file', type=str, action='store',
                        help='The path to the raw data file.')
    args = parser.parse_args()

    main(args.raw_data_file)
