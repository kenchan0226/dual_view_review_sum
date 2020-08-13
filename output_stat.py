import os
from os.path import join
import argparse
import re
from collections import Counter, defaultdict
import numpy as np


def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.dec')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def count_lines_and_tokens(fname):
    num_tokens = 0
    one_gram_counter = Counter()
    two_gram_counter = Counter()
    three_gram_counter = Counter()
    with open(fname) as f:
        for i, l in enumerate(f):
            l_tokenized = l.strip().split(" ")
            one_gram_counter.update(_make_n_gram(l_tokenized, n=1))
            two_gram_counter.update(_make_n_gram(l_tokenized, n=2))
            three_gram_counter.update(_make_n_gram(l_tokenized, n=3))
            num_tokens += len(l_tokenized)
    one_gram_repeat = sum(c - 1 for g, c in one_gram_counter.items() if c > 1)
    two_gram_repeat = sum(c - 1 for g, c in two_gram_counter.items() if c > 1)
    three_gram_repeat = sum(c - 1 for g, c in three_gram_counter.items() if c > 1)
    three_gram_repeat = sum(c - 1 for g, c in three_gram_counter.items() if c > 1)
    return i + 1, num_tokens, one_gram_repeat, two_gram_repeat, three_gram_repeat


def main(args):
    output_path = join(args.decode_dir, "output")
    n_output = _count_data(output_path)
    total_num_sentences = 0
    total_num_tokens = 0
    one_gram_repeat_sum = 0
    two_gram_repeat_sum = 0
    three_gram_repeat_sum = 0
    num_tokens_list = []

    for i in range(n_output):  # iterate every .dec
        dec_file_path = join(output_path, "{}.dec".format(i))
        num_sentences, num_tokens, one_gram_repeat, two_gram_repeat, three_gram_repeat = count_lines_and_tokens(dec_file_path)
        total_num_sentences += num_sentences
        total_num_tokens += num_tokens
        one_gram_repeat_sum += one_gram_repeat
        two_gram_repeat_sum += two_gram_repeat
        three_gram_repeat_sum += three_gram_repeat
        num_tokens_list.append(num_tokens)

    print("average generated sentences: {:.3f}".format(total_num_sentences/n_output))
    print("average tokens per sentence: {:.3f}".format(total_num_tokens/total_num_sentences))
    print("average repeat 1-gram: {:.3f}".format(one_gram_repeat_sum / n_output))
    print("average repeat 2-gram: {:.3f}".format(two_gram_repeat_sum / n_output))
    print("average repeat 3-gram: {:.3f}".format(three_gram_repeat_sum / n_output))
    num_tokens_array = np.array(num_tokens_list)
    print("min tokens: {}".format(num_tokens_array.min()))
    print("max tokens: {}".format(num_tokens_array.max()))
    print("std of tokens: {}".format(np.std(num_tokens_array)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Output statistics')

    # choose metric to evaluate
    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')
    args = parser.parse_args()
    main(args)
