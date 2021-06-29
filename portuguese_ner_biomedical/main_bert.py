__author__='thiagocastroferreira'

import torch
import torch.nn as nn
from torch import optim
import os
import time

from model import BERTSlotFilling
from dataset import DataBERT
from trainer import Trainer
from evaluation import Evaluation

OUTPUT_PATH = 'output_files/'
NUM_EXPERIMENTS = 5
NUM_EPOCHS = 1  # The number of epochs (full passes through the data) to train for
BATCH = 16
HIDDEN_DIM = 1024
FOLDER_PATH = 'data'

def main():
    # create output folder
    time_str = time.strftime("%Y_%m_%d-%H:%M:%S")
    output_folder = OUTPUT_PATH + ('linear_layer_crf_%s' % time_str) + '/'
    os.mkdir(output_folder)

    data_info = DataBERT(FOLDER_PATH)
    train_data, test_data = data_info.fit()

    # Training settings
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    num_classes = len(data_info.vocab_out)

    model = BERTSlotFilling(HIDDEN_DIM, num_classes)
    model.to(device)

    evaluation = Evaluation(output_folder)
    trainer = Trainer(model)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=1e-5)
    weights = [1.] * num_classes
    weights[data_info.out_w2id['O']] = 0.01
    weights = torch.tensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    micro_avg_f1 = 0.0
    y_true_text = y_pred_text = test_tokens = []
    for num_experiment in range(NUM_EXPERIMENTS):
        y_true, y_pred = trainer.test_bert(test_data)
        for epoch in range(1, NUM_EPOCHS + 1):
            trainer.train_bert(train_data, criterion, optimizer, BATCH, epoch)
            y_true, y_pred = trainer.test_bert(test_data)

        # get test tokens and convert output from number to text
        test_tokens = [info[-1] for info in test_data]
        y_true_text = evaluation.convert_output_to_text(y_true, data_info.out_id2w)
        y_pred_text = evaluation.convert_output_to_text(y_pred, data_info.out_id2w)

        micro_avg_f1 += evaluation.evaluate(num_experiment, y_true_text, y_pred_text)

    micro_avg_f1 /= NUM_EXPERIMENTS
    print()
    print('Micro avg F1: %.2f' % micro_avg_f1)
    evaluation.generate_output_csv('bert_output', y_true_text, y_pred_text, test_tokens)


if __name__ == "__main__":
    main()
