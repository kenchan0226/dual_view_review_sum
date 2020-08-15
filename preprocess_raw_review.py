"""Split into train, val, test. Append period to summary. Filter out too short summary and review. Build vocab"""

from os.path import join
import json
import os
import random
import argparse
from collections import Counter
import pickle as pkl

END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', ")"]
#MIN_SUM_LEN = 6
#MIN_REVIEW_LEN = 21
MAX_REVIEW_LEN = 800


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."


def main(tokenized_data_file, out_dir, num_valid, num_test, is_shuffle, min_review_len, min_sum_len):
    assert not os.path.exists(out_dir)
    os.makedirs(join(out_dir))
    for split in ['train', 'val', 'test']:
        os.makedirs(join(out_dir, split))

    num_processed_review_valid = 0
    num_processed_review_test = 0
    num_processed_review_train = 0
    num_removed_samples = 0
    num_short_reviews = 0
    num_short_summaries = 0
    num_long_reviews = 0
    vocab_counter = Counter()

    with open(tokenized_data_file) as f_in:
        print("Reading {}".format(tokenized_data_file))
        all_lines = f_in.readlines()
        idx_list = list(range(len(all_lines)))
        if is_shuffle:
            idx_and_lines = list(zip(idx_list, all_lines))
            random.shuffle(idx_and_lines)
            idx_list, all_lines = zip(*idx_and_lines)
        print("Finish Reading")
        for review_i, line in enumerate(all_lines):
            if (review_i + 1) % 10000 == 0:
                print("Processed {} samples".format(review_i+1))

            js = json.loads(line.strip())

            summary = js['summary']
            summary_fixed_period = [fix_missing_period(l) for l in summary]
            js['summary'] = summary_fixed_period
            summary_text = ' '.join(summary_fixed_period).strip()
            summary_word_list = summary_text.strip().split(' ')

            review = js['reviewText']
            if len(review) > 0:
                review[-1] = fix_missing_period(review[-1])
            review_text = ' '.join(review).strip()
            review_word_list = review_text.strip().split(' ')

            # skip empty review
            if len(summary_text) == 0 or len(review_text) == 0 or js['overall'] == "":
                num_removed_samples += 1
                continue
            if len(summary_word_list) < min_sum_len:
                num_removed_samples += 1
                num_short_summaries += 1
                continue
            if len(review_word_list) < min_review_len:
                num_removed_samples += 1
                num_short_reviews += 1
                continue
            if len(review_word_list) > MAX_REVIEW_LEN:
                num_removed_samples += 1
                num_long_reviews += 1
                continue

            # choose split
            if num_processed_review_valid < num_valid:
                split_out_dir = join(out_dir, 'val')
                json_id = num_processed_review_valid
                num_processed_review_valid += 1
            elif num_processed_review_test < num_test:
                split_out_dir = join(out_dir, 'test')
                json_id = num_processed_review_test
                num_processed_review_test += 1
            else:
                split_out_dir = join(out_dir, 'train')
                json_id = num_processed_review_train
                num_processed_review_train += 1
                # update vocab
                all_tokens = summary_word_list + review_word_list
                vocab_counter.update([t for t in all_tokens if t != ""])

            with open(join(split_out_dir, '{}.json'.format(json_id)), 'w') as f_out:
                json.dump(js, f_out, indent=4)

    print("training samples:\t{}".format(num_processed_review_train))
    print("validation samples:\t{}".format(num_processed_review_valid))
    print("test samples:\t{}".format(num_processed_review_test))

    print("num_removed_samples:\t{}".format(num_removed_samples))
    print("num_short_reviews:\t{}".format(num_short_reviews))
    print("num_long_reviews:\t{}".format(num_long_reviews))
    print("num_short_summaries:\t{}".format(num_short_summaries))

    print("Writing vocab file....")
    with open(os.path.join(out_dir, "vocab_cnt.pkl"),
              'wb') as vocab_file:
        pkl.dump(vocab_counter, vocab_file)
    print("Finished writing vocab file. \nNumber of vocab:\t{}".format(len(vocab_counter)))

    print("Writing split idx")
    val_indices = list(idx_list[:num_valid])
    test_indices = list(idx_list[num_valid:num_valid+num_test])
    #train_indices = list(idx_list[num_valid+num_test:])
    with open(os.path.join(out_dir, "val_indices.txt"),
              'w') as val_indices_file:
        val_indices_file.write(','.join([str(i) for i in val_indices]) + '\n')
    with open(os.path.join(out_dir, "test_indices.txt"),
              'w') as test_indices_file:
        test_indices_file.write(','.join([str(i) for i in test_indices]) + '\n')
    #with open(os.path.join(out_dir, "train_indices.txt"),
    #          'wb') as train_indices_file:
    #    train_indices_file.write(','.join([str(i) for i in train_indices]))
    print("Finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=('Preprocess review data')
    )
    parser.add_argument('-tokenized_data_file', type=str, action='store',
                        help='The path to the tokenized data file.')
    parser.add_argument('-out_dir', type=str, action='store',
                        help='The directory of the output data.')
    parser.add_argument('-num_valid', type=int, action='store',
                        help='Number of validation samples.')
    parser.add_argument('-num_test', type=int, action='store',
                        help='Number of test samples.')
    parser.add_argument('-is_shuffle', action='store_true',
                        help='Randomly shuffle the training data.')
    parser.add_argument('-seed', type=int, action='store', default=9527,
                        help='random seed.')
    parser.add_argument('-min_review_len', type=int, action='store', default=16,
                        help='Minimum tokens in review.')
    parser.add_argument('-min_summary_len', type=int, action='store', default=4,
                        help='Minimum tokens in summar.')

    args = parser.parse_args()

    random.seed(args.seed)

    main(args.tokenized_data_file, args.out_dir, args.num_valid, args.num_test, args.is_shuffle, args.min_review_len, args.min_summary_len)
