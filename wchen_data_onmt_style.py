import os
import json
from tqdm import tqdm
data_path_base = "/research/king3/ik_grp/wchen/amazon_review_data/"
saved_data_path_base = "/research/king3/ik_grp/wchen/amazon_review_data/onmt_style_data/"


def get_onmt_style_data(dataset='cloth'):
    dataset_map = {'cloth': 'min_4_Clothing_Shoes_and_Jewelry_5',
                   'home': 'min_4_Home_and_Kitchen_5',
                   'sports': 'min_4_reviews_Sports_and_Outdoors_5',
                   'toys': 'min_4_Toys_and_Games_5',
                   'movies': 'min_4_Movies_and_TV_5'}
    assert dataset in dataset_map
    for split_type in ['test', 'val', 'train']:
        data_path = os.path.join(data_path_base, dataset_map[dataset], split_type)
        print("Processing ", data_path)
        saved_data_path = os.path.join(saved_data_path_base, dataset)
        if not os.path.exists(saved_data_path):
            os.makedirs(saved_data_path)
        saved_review_file = open(os.path.join(saved_data_path, '{}_{}_review.txt'.format(dataset, split_type)), 'w', encoding='utf-8')
        saved_summary_file = open(os.path.join(saved_data_path, '{}_{}_summary.txt'.format(dataset, split_type)), 'w', encoding='utf-8')

        json_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        file_cnt = len(json_files)
        for i in tqdm(range(file_cnt)):
            f = os.path.join(data_path, '{}.json'.format(i))
            f = open(f, encoding='utf-8')
            data_i = json.load(f)
            review_line = ' '.join(data_i['reviewText']) + '\n'
            summary_line = ' '.join(data_i['summary']) + '\n'
            saved_review_file.write(review_line)
            saved_summary_file.write(summary_line)
        print("Reviews are saved to {}".format(os.path.join(saved_data_path, '{}_{}_review.txt'.format(dataset, split_type))))
        print("Summaries are saved to {}".format(os.path.join(saved_data_path, '{}_{}_summary.txt'.format(dataset, split_type))))


if __name__ == "__main__":
    get_onmt_style_data('cloth')
    get_onmt_style_data('home')
    get_onmt_style_data('sports')
    get_onmt_style_data('toys')
    get_onmt_style_data('movies')




