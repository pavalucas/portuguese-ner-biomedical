__author__ = 'lucasaguiarpavanelli'

from torch import optim
import os
import time

from model import LinearLayerCRF
from dataset import Data
from trainer import Trainer
from evaluation import Evaluation

OUTPUT_PATH = 'output_files/'
NUM_EXPERIMENTS = 5
NUM_EPOCHS = 10
BATCH = 16
FOLDER_PATH = 'data'


def main():
    # create output folder
    time_str = time.strftime("%Y_%m_%d-%H:%M:%S")
    output_folder = OUTPUT_PATH + ('linear_layer_crf_%s' % time_str) + '/'
    os.mkdir(output_folder)

    data_info = Data(FOLDER_PATH)
    train_data, test_data = data_info.fit()
    vocab_size = len(data_info.vocab_in)
    num_classes = len(data_info.vocab_out)

    model = LinearLayerCRF(num_classes, vocab_size, data_info.out_w2id)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    trainer = Trainer(model)
    evaluation = Evaluation(output_folder)

    micro_avg_f1 = 0.0
    y_true_text = y_pred_text = test_tokens = []
    for num_experiment in range(NUM_EXPERIMENTS):
        y_true, y_pred = trainer.test_linear_layer_crf(test_data)
        for epoch in range(1, NUM_EPOCHS + 1):
            trainer.train_linear_layer_crf(train_data, optimizer, BATCH, epoch)
            y_true, y_pred = trainer.test_linear_layer_crf(test_data)

        # get test tokens and convert output from number to text
        test_tokens = [info[-1] for info in test_data]
        y_true_text = evaluation.convert_output_to_text(y_true, data_info.out_id2w)
        y_pred_text = evaluation.convert_output_to_text(y_pred, data_info.out_id2w)

        micro_avg_f1 += evaluation.evaluate(num_experiment, y_true_text, y_pred_text)

    micro_avg_f1 /= NUM_EXPERIMENTS
    print()
    print('Micro avg F1: %.2f' % micro_avg_f1)
    evaluation.generate_output_csv('linear_layer_output', y_true_text, y_pred_text, test_tokens)


if __name__ == "__main__":
    main()