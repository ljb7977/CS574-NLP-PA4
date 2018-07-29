import numpy

import loader
import model
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F


def my_collate(batch):
    tokens = [item['tokens'] for item in batch]
    pos1 = [item['pos1'] for item in batch]
    pos2 = [item['pos2'] for item in batch]
    rel = [item['rel'] for item in batch]
    return [tokens, pos1, pos2, rel]


class Trainer():
    def __init__(self, train_path, test_path, learning_rate=0.001):
        self.train_set = loader.TextDataset(train_path)
        self.test_set = loader.TextDataset(test_path, word_vocab=self.train_set.word2idx,
                                           pos_vocab=self.train_set.pos2idx)

        self.labels = {
            'country': 0, 'team': 1,
            'starring': 2, 'director': 3,
            'child': 4, 'successor': 5
        }

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
        self.learningRate = learning_rate
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size,
                                                        shuffle=True, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=1,
                                                       shuffle=False, num_workers=4)

    def epoch(self, criterion, optimizer):
        self.cnn_model.train()
        for i, data in enumerate(self.train_loader, 0):
            tokens, pos1, pos2, labels = data['tokens'], data['pos1'], data['pos2'], data['rel']
            # print(tokens, pos1, pos2, labels)
            optimizer.zero_grad()
            logits = self.cnn_model(tokens, pos1, pos2)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            print("step: ", i, " loss: ", round(loss.item(), 2))

    def train(self, mode, epochs=5):
        optimizer = optim.Adam(self.cnn_model.parameters(), lr=self.learningRate, weight_decay=0.05, amsgrad=True)
        criterion = nn.CrossEntropyLoss()
        for i in range(epochs):
            print("Epoch " + str(i))
            self.epoch(criterion, optimizer)
            self.test(epoch=i)

            torch.save(self.cnn_model, "model/"+str(mode)+"/epoch_"+str(i)+".model")

    def test(self, epoch=0, load=False, model_path=None):
        print("Test start for ", model_path)
        if load:
            self.cnn_model = torch.load(model_path)

        self.cnn_model.eval()
        result = []
        with torch.no_grad():
            count = 0
            correct = 0
            for data in self.test_loader:
                count += 1
                tokens, pos1, pos2, labels = data['tokens'], data['pos1'], data['pos2'], data['rel']
                logits = F.softmax(self.cnn_model(tokens, pos1, pos2), dim=1)
                result.append((data['sbj'], data['obj'], data['rel'].item(), logits.squeeze()))
                _, ans =  torch.max(logits, dim=1)
                if data['rel'].item() == ans.item():
                    correct+=1

        print("Analysis start: ", epoch)
        gold_set = set()
        for dat in self.test_set.data:
            gold_set.add((dat['sbj'], dat['obj'], dat['rel']))
        # print(gold_set)

        with open("analysis_" + str(epoch) + ".txt", "w") as output:
            print("model: ", epoch, file=output)
            print("threshold\trecall\tprecision", file=output)

            for threshold in numpy.arange(0.05, 1, 0.05):
                predict_set = set()

                for r in result:
                    sbj, obj, rel, logits = r
                    indices = numpy.where(logits > threshold)[0].tolist()
                    # print(indices)
                    # print([(sbj[0], obj[0], i) for i in indices])
                    predict_set.update([(sbj[0], obj[0], i) for i in indices])

                correct_set = predict_set & gold_set

                if len(predict_set) == 0:
                    precision = 0
                else:
                    precision = len(correct_set)/len(predict_set)
                recall = len(correct_set) / len(gold_set)

                print(round(threshold, 3), round(recall, 3), round(precision, 3), file=output)
        print("Analysis Done")
