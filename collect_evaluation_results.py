import os
import csv
import argparse


def collect_train_test_evaluation_results(args):
    training_log_file = args.training_log_file
    testing_log_file = args.testing_log_file
    model_name = args.model_name

    if len(training_log_file) != 0:
        csv_saved_dir = os.path.dirname(training_log_file)
        assert model_name in training_log_file, "The specified model name should in the path of training log file!"
    if len(testing_log_file) != 0:
        csv_saved_dir = os.path.dirname(testing_log_file)
        assert model_name in testing_log_file, "The specified model name should in the path of testing log file!"

    saved_csv = os.path.join(csv_saved_dir, "{}.csv".format(model_name))
    print("Evaluation results are saved in {}".format(saved_csv))

    fieldnames = ["model_name", "model path", "predict path",
                  "valid joint loss", "valid ppl",
                  "valid class total loss", "valid class total f1",
                  "valid class enc loss", "valid class enc f1",
                  "valid class dec loss", "valid class dec f1",
                  "Rouge 1 R", "Rouge 1 P", "Rouge 1 F",
                  "Rouge 2 R", "Rouge 2 P", "Rouge 2 F",
                  "Rouge L R", "Rouge L P", "Rouge L F",
                  "micro F1", "Accuracy", "balanced_enc_acc",
                  "dec micro F1", "dec Accuracy", "balanced_dec_acc",
                  "merged micro F1", "merged Accuracy", "balanced_merged_acc"]

    with open(saved_csv, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        rslt_dict = {}
        for k in fieldnames:
            rslt_dict[k] = ''

        rslt_dict["model_name"] = model_name

        if os.path.exists(training_log_file):
            log_file = open(training_log_file, encoding='utf-8')
            for line in log_file:
                if 'final_best_valid_joint_loss:' in line:
                    rslt_dict['valid joint loss'] = line.split('final_best_valid_joint_loss:')[-1].strip()
                if 'final_correspond_valid_ppl:' in line:
                    rslt_dict['valid ppl'] = line.split('final_correspond_valid_ppl:')[-1].strip()
                if 'final_correspond_class_loss:' in line:
                    rslt_dict['valid class loss'] = line.split('final_correspond_class_loss:')[-1].strip()
                if 'final_correspond_class_f1:' in line:
                    rslt_dict['valid class f1'] = line.split('final_correspond_class_f1:')[-1].strip()
                if 'final_correspond_enc_class_loss:' in line:
                    rslt_dict['valid class enc loss'] = line.split('final_correspond_enc_class_loss:')[-1].strip()
                if 'final_correspond_enc_class_f1:' in line:
                    rslt_dict['valid class enc f1'] = line.split('final_correspond_enc_class_f1:')[-1].strip()
                if 'final_correspond_dec_class_loss:' in line:
                    rslt_dict['valid class dec loss'] = line.split('final_correspond_dec_class_loss:')[-1].strip()
                if 'final_correspond_dec_class_f1:' in line:
                    rslt_dict['valid class dec f1'] = line.split('final_correspond_dec_class_f1:')[-1].strip()
            log_file.close()

        if os.path.exists(testing_log_file):
            log_file = open(testing_log_file, encoding='utf-8')
            for line in log_file:
                for len_str in ['1', '2', 'L']:
                    for mode in ['R', 'P', 'F']:
                        splitter = "1 ROUGE-{} Average_{}:".format(len_str, mode)
                        if splitter in line:
                            rslt = line.split(splitter)[-1].strip()
                            rslt = rslt.split(' ')[0].strip()
                            rslt_dict['Rouge {} {}'.format(len_str, mode)] = rslt

                if 'micro f1 score:' in line[:len('micro f1 score:')]:
                    rslt_dict['micro F1'] = line.split('micro f1 score:')[-1].strip()
                if 'accuracy:' in line[:len('accuracy:')]:
                    rslt_dict['Accuracy'] = line.split('accuracy:')[-1].strip()
                if 'balanced_enc_acc:' in line:
                    rslt_dict['balanced_enc_acc'] = line.split('balanced_enc_acc:')[-1].strip()
                if 'dec micro f1 score:' in line:
                    rslt_dict['dec micro F1'] = line.split('dec micro f1 score:')[-1].strip()
                if 'dec accuracy:' in line:
                    rslt_dict['dec Accuracy'] = line.split('dec accuracy:')[-1].strip()
                if 'balanced_dec_acc:' in line:
                    rslt_dict['balanced_dec_acc'] = line.split('balanced_dec_acc:')[-1].strip()
                if 'merged micro f1 score:' in line:
                    rslt_dict['merged micro F1'] = line.split('merged micro f1 score:')[-1].strip()
                if 'merged accuracy:' in line:
                    rslt_dict['merged Accuracy'] = line.split('merged accuracy:')[-1].strip()
                if 'balanced_merged_acc:' in line:
                    rslt_dict['balanced_merged_acc'] = line.split('balanced_merged_acc:')[-1].strip()
        writer.writerow(rslt_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("arguments for collecting evaluation results")
    parser.add_argument('-training_log_file', type=str, default='', help='path to the training log file')
    parser.add_argument('-testing_log_file', type=str, default='', help='path to the testing log file')
    parser.add_argument('-model_name', type=str, default='', help='the name of the evaluated model')

    args = parser.parse_args()
    collect_train_test_evaluation_results(args)
