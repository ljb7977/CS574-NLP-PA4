import loader
import model
from torch import optim
import torch
import torch.nn as nn

class Trainer():
    def __init__(self, train_path = "data/ds_train.tsv", learningRate=0.001):
        self.data, word2idx, pos2idx = loader.load_data_and_labels(train_path)
        args = {
            'pos_embed_dim': 256,
            'word_embed_dim': 256,
            'words_num': len(word2idx),
            'pos_num': len(pos2idx),
            'kernel_num': 32,
            'kernel_sizes': [2, 3, 4, 5]
        }
        batch_size = 4
        self.cnn_model = model.CNN_Text(args)
        self.learningRate = learningRate

        self.loader = torch.utils.data.DataLoader(self.data, batch_size=batch_size,
                                                  shuffle=True, num_workers=4)

    def epoch(self, criterion, optimizer):
        for i, data in enumerate(self.loader, 0):
            labels, sentences = data
            optimizer.zero_grad()
            logits = self.cnn_model(sentences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            print("loss: ", round(loss.item(), 2))

    def train(self, epochs=5):
        optimizer = optim.Adam(self.cnn_model.parameters(), lr=self.learningRate, weight_decay=0.05, amsgrad=True)
        criterion = nn.CrossEntropyLoss()
        for i in range(epochs):
            print("Epoch " + str(i))
            self.epoch(criterion, optimizer)