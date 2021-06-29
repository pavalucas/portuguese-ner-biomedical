__author__='lucasaguiarpavanelli'

import torch


class Trainer:
    def __init__(self, model):
        self.model = model

    def train_linear_layer_crf(self, train_data, optimizer, batch, epoch):
        self.model.train()
        loss, losses = 0, 0
        for i, (token_ids, tag_ids, _) in enumerate(train_data):
            # Init
            optimizer.zero_grad()

            # Calculate loss
            l = self.model.loss(token_ids, tag_ids)
            loss += l
            losses += l

            if (i + 1) % batch == 0:
                # Backpropagation
                loss = loss / batch
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Display
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, losses / i), end='\r')
                loss = 0

        print()

    def train_bert(self, train_data, criterion, optimizer, batch, epoch, device='cpu'):
                   #device='cuda'):
        self.model.train()
        loss, losses = 0, 0
        for i, (token_ids, subwords_idx, tag_ids) in enumerate(train_data):
            # Init
            optimizer.zero_grad()

            # Predict
            output = self.model(token_ids, subwords_idx)

            # Calculate loss
            l = criterion(output, tag_ids.to(device))
            loss += l
            losses += l

            if (i + 1) % batch == 0:
                # Backpropagation
                loss = loss / batch
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Display
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, losses / i), end='\r')
                loss = 0

        print()

    def test_linear_layer_crf(self, test_data):
        self.model.eval()
        y_pred, y_true = [], []
        for i, (token_ids, tag_ids, _) in enumerate(test_data):
            output = self.model(token_ids)
            y_pred.append([int(w) for w in list(output[0])])
            y_true.append([int(w) for w in list(tag_ids[0])])
        return y_true, y_pred

    def test_bert(self, test_data):
        self.model.eval()
        y_pred, y_true = [], []
        for i, (token_ids, subwords_idx, tag_ids) in enumerate(test_data):
            output = self.model(token_ids, subwords_idx)
            y_pred.append([int(w) for w in list(torch.argmax(output, dim=1))])
            y_true.append([int(w) for w in list(tag_ids)])
        return y_true, y_pred
