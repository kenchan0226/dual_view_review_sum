import os
import argparse
import pickle as pkl


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)


def vocab_jaccard_sim(topk_vocabs):
    num = len(topk_vocabs)
    sims = []
    for i in range(num):
        sims_i = []
        for j in range(num):
            sims_i_j = jaccard_similarity(topk_vocabs[i], topk_vocabs[j])
            sims_i.append(round(sims_i_j, 3))
        sims.append(sims_i)
    for sim in sims:
        print(sim)
    """
    [1.0, 0.544, 0.413, 0.333, 0.316]
    [0.544, 1.0, 0.619, 0.429, 0.351]
    [0.413, 0.619, 1.0, 0.581, 0.439]
    [0.333, 0.429, 0.581, 1.0, 0.646]
    [0.316, 0.351, 0.439, 0.646, 1.0]
    One thought: consider to remove the overlapped tokens
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser("rating vocab statistics")
    parser.add_argument('-data_dir', type=str, help="The directory of the data")
    parser.add_argument('-topk', type=int, default=200, help="Calculate the statistics of the top 200 words of each rating")
    args = parser.parse_args()

    # load the word_counter
    topk_vocab_tokens = []
    for i in range(1, 6):
        vocab_i = os.path.join(args.data_dir, 'rating_{}_vocab_counter_no_stop_word_and_punc.pkl'.format(i))
        vocab_i = pkl.load(open(vocab_i, 'rb'))
        vocab_i = vocab_i.most_common(args.topk)
        vocab_tokens_i = [w[0] for w in vocab_i]
        topk_vocab_tokens.append(vocab_tokens_i)

    # calculate the jaccard similarities
    vocab_jaccard_sim(topk_vocab_tokens)