__author__='lucasaguiarpavanelli'

import pandas as pd
from sklearn.metrics import accuracy_score
from seqeval.metrics import classification_report


class Evaluation:
    def __init__(self, output_folder):
        self.output_folder = output_folder

    def convert_output_to_text(self, y, out_id2w):
        result = []
        for indexes in y:
            result.append([out_id2w[index] for index in indexes])
        return result

    def get_single_output_id_list(self, y):
        return [index for indexes in y for index in indexes]

    def evaluate(self, num_experiment, y_true, y_pred):
        """
        Evaluate model's predictions using classification_report and return micro avg f1-score
        """
        # y_true_id = get_single_output_id_list(y_true)
        # y_pred_id = get_single_output_id_list(y_pred)
        # y_baseline = len(y_true_id) * [data_info.out_w2id['O']]
        # print('Accuracy Score: ', accuracy_score(y_true_id, y_pred_id))
        # print('Accuracy Score Baseline: ', accuracy_score(y_true_id, y_baseline))
        print(classification_report(y_true, y_pred))
        class_report = classification_report(y_true, y_pred, output_dict=True)
        df = pd.DataFrame(class_report).transpose()
        df.to_csv(f'{self.output_folder}classification_report_experiment_{num_experiment}.csv', sep=';')
        return class_report['micro avg']['f1-score']

    def generate_output_csv(self, file_name, y_true, y_pred, test_tokens):
        csv_dict = {'id': [], 'token': [], 'true_tag': [], 'pred_tag': []}
        test_id = 0
        for cur_tokens, cur_y_true, cur_y_pred in zip(test_tokens, y_true, y_pred):
            for token, true_tag, pred_tag in zip(cur_tokens, cur_y_true, cur_y_pred):
                csv_dict['id'].append(test_id)
                csv_dict['token'].append(token)
                csv_dict['true_tag'].append(true_tag)
                csv_dict['pred_tag'].append(pred_tag)
            test_id += 1
        df = pd.DataFrame(csv_dict, columns=['id', 'token', 'pred_tag', 'true_tag'])
        df.to_csv(f'{self.output_folder}{file_name}.csv', index=False, sep=';')