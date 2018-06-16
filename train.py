import loader
import model
from torch import optim
import torch
import torch.nn as nn

class Trainer():
    def __init__(self, train_path = "data/ds_train.tsv", learningRate=0.001, epochs=10):
        self.data, word2idx, pos2idx = loader.load_data_and_labels(train_path)
        args = {
            'pos_embed_dim': 256,
            'word_embed_dim': 256,
            'words_num': len(word2idx),
            'pos_num': len(pos2idx),
            'kernel_num': 5,
            'kernel_sizes': [64, 64, 64, 64, 64]
        }
        self.cnn_model = model.CNN_Text(args)
        optimizer = optim.Adam(self.cnn_model.parameters(), lr=learningRate, weight_decay=0.05, amsgrad=True)
        criterion = nn.CrossEntropyLoss()

        for i in range(epochs):
            print("Epoch " + str(i))
            self.train(criterion, optimizer)

    def train(self, criterion, optimizer):
        optimizer.zero_grad()
        logits = self.cnn_model(self.data)

        loss = criterion(logits, gold)
        optimizer.step()
        print("loss: ", round(loss.item(), 2))

        return
