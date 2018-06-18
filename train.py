import loader
import model
from torch import optim
import torch
import torch.nn as nn


def my_collate(batch):
    tokens = [item['tokens'] for item in batch]
    pos1 = [item['pos1'] for item in batch]
    pos2 = [item['pos2'] for item in batch]
    rel = [item['rel'] for item in batch]
    return [tokens, pos1, pos2, rel]

class Trainer():
    def __init__(self, train_path = "data/ds_train.tsv",
                 test_path = "data/gold_test.tsv", learningRate=0.001):
        self.train_set = loader.TextDataset(train_path)
        self.test_set = loader.TextDataset(test_path, word_vocab = self.train_set.word2idx,
                                           pos_vocab = self.train_set.pos2idx)
        args = {
            'pos_embed_dim': 256,
            'word_embed_dim': 256,
            'words_num': len(self.train_set.word2idx),
            'pos_num': len(self.train_set.pos2idx),
            'kernel_num': 32,
            'kernel_sizes': [2, 3, 4, 5],
            'dropout': 0.2,
            'static': False
        }
        batch_size = 4
        self.cnn_model = model.CNN_Text(args)
        self.learningRate = learningRate
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size,
                                                        shuffle=True, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=1,
                                                       shuffle=False, num_workers=4)

    def epoch(self, criterion, optimizer):
        for i, data in enumerate(self.train_loader, 0):
            tokens, pos1, pos2, labels = data['tokens'], data['pos1'], data['pos2'], data['rel']
            # print(tokens, pos1, pos2, labels)
            optimizer.zero_grad()
            logits = self.cnn_model(tokens, pos1, pos2)
            # print(logits)
            # print(labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            print("step: ", i, " loss: ", round(loss.item(), 2))

    def train(self, epochs=5):
        optimizer = optim.Adam(self.cnn_model.parameters(), lr=self.learningRate, weight_decay=0.05, amsgrad=True)
        criterion = nn.CrossEntropyLoss()
        for i in range(epochs):
            print("Epoch " + str(i))
            self.epoch(criterion, optimizer)
            self.validate(epoch=i)

    def validate(self, epoch):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.train_loader:
                tokens, pos1, pos2, labels = data['tokens'], data['pos1'], data['pos2'], data['rel']
                logits = self.cnn_model(tokens, pos1, pos2)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("on epoch ", epoch, ", Accuracy on dev set: ", 100*correct/total)

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                tokens, pos1, pos2, labels = data['tokens'], data['pos1'], data['pos2'], data['rel']
                logits = self.cnn_model(tokens, pos1, pos2)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Accuracy on test set: ", 100 * correct / total)