import csv
import os
import argparse
from scipy import stats


def main(args):
    rg_keys = ['Rouge 1 R', 'Rouge 1 P', 'Rouge 1 F',
               'Rouge 2 R', 'Rouge 2 P', 'Rouge 2 F',
               'Rouge L R', 'Rouge L P', 'Rouge L F']
    acc_keys = ['enc_macro_F1', 'enc_Accuracy', 'balanced_enc_acc']
    merged_keys = rg_keys + acc_keys
    # merged_keys = rg_keys

    # seeds = [9527, 556, 3435, 1001, 520]
    seeds = [1001, 520, 2, 250, 610]

    # bs_models = ['SingleMultiHopClassifier', 'SingleMaxClassifier', 'HSSC', 'Max', 'PGNet', 'HSSC_copy', 'Max_copy']
    bs_models = ['SingleMultiHopClassifier', 'HSSC', 'Max', 'PGNet', 'HSSC_copy', 'Max_copy']
    our_models = ['Multi_view']

    bs_models_dict = {}
    for bs_m in bs_models:
        bs_models_dict[bs_m] = {}
        for key in merged_keys:
            bs_models_dict[bs_m][key] = []

    our_models_dict = {}
    for our_m in our_models:
        our_models_dict[our_m] = {}
        for key in merged_keys:
            our_models_dict[our_m][key] = []

    exp_resilts_file_name = args.exp_results_csv
    assert '.csv' in exp_resilts_file_name
    csv_file = csv.DictReader(open(os.path.join(exp_resilts_file_name), encoding='utf-8'))

    saved_file_name = exp_resilts_file_name.split('.csv')[0] + '_p_values.csv'
    saved_fieldnames = ['model_name'] + merged_keys
    saved_csv_file = csv.DictWriter(open(os.path.join(saved_file_name), 'w', encoding='utf-8'),
                                    fieldnames=saved_fieldnames)
    saved_csv_file.writeheader()

    for row in csv_file:

        if len(row['model_name'].strip()) == 0:
            continue

        model_name, seed = row['model_name'].split('_seed', 1)
        seed = int(seed.strip())
        assert seed in seeds

        if model_name in bs_models_dict:
            for key in merged_keys:
                value = round(float(row[key]), 4)
                bs_models_dict[model_name][key].append(value)
        else:
            assert model_name in our_models_dict
            for key in merged_keys:
                value = round(float(row[key]), 4)
                our_models_dict[model_name][key].append(value)

    for our_m in our_models:
        for bs_m in bs_models:
            out_line = "{} v.s. {}:".format(our_m, bs_m)
            saved_p_values = {'model_name': "{} v.s. {}:".format(our_m, bs_m)}
            for key in merged_keys:
                bs_results = bs_models_dict[bs_m][key]
                assert len(seeds) == len(bs_results)
                our_results = our_models_dict[our_m][key]
                assert len(seeds) == len(our_results)

                p = stats.ttest_rel(bs_results, our_results).pvalue.tolist()
                p = round(p, 3)
                out_line = out_line + " {}:{}".format(key, p)
                saved_p_values[key] = p
            print(out_line)
            saved_csv_file.writerow(saved_p_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Paired t-test for experimental results.')
    parser.add_argument('-exp_results_csv', action='store', required=True, help='directory of decoded summaries')
    args = parser.parse_args()
    main(args)