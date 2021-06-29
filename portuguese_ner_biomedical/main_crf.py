import sklearn_crfsuite
import random
import os
import time

from dataset import Data
from model import CRF
from evaluation import Evaluation

OUTPUT_PATH = 'output_files/'
DATA_PATH = 'data_2021_02_23'
NUM_EXPERIMENTS = 5


def main():
    # create output folder
    time_str = time.strftime("%Y_%m_%d-%H:%M:%S")
    output_folder = OUTPUT_PATH + ('crf_%s' % time_str) + '/'
    os.mkdir(output_folder)

    data_info = Data(DATA_PATH)
    crf = CRF()
    evaluation = Evaluation(output_folder)

    micro_avg_f1 = 0.0
    y_true = y_pred = test_tokens = []
    for num_experiment in range(NUM_EXPERIMENTS):
        x_train, y_train, x_test, y_true, test_tokens = crf.get_train_test_data(data_info)
        crf.fit(x_train, y_train)
        y_pred = crf.predict(x_test)

        micro_avg_f1 += evaluation.evaluate(num_experiment, y_true, y_pred)

        # y_true_id = [data_info.out_w2id[term] for example in y_true for term in example]
        # y_pred_id = [data_info.out_w2id[term] for example in y_pred for term in example]
        # y_baseline = len(y_true_id) * [data_info.out_w2id['O']]
        # print('Accuracy Score CRF: ', accuracy_score(y_true_id, y_pred_id))
        # print('Accuracy Score Baseline: ', accuracy_score(y_true_id, y_baseline))
    micro_avg_f1 /= NUM_EXPERIMENTS
    print()
    print('Micro avg F1: %.2f' % micro_avg_f1)

    evaluation.generate_output_csv('crf_output', y_true, y_pred, test_tokens)


if __name__ == "__main__":
    main()
