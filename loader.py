import torch
from konlpy.tag import Hannanum
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, filepath, word_vocab=None, pos_vocab=None):
        self.max_sentence_length = 150
        self.labels = {
            'country': 0, 'team': 1,
            'starring': 2, 'director': 3,
            'child': 4, 'successor': 5
        }

        data = self.parse_line(filepath)

        if word_vocab is None:
            self.word2idx, self.pos2idx = self.make_vocab(data)
        else:
            self.word2idx, self.pos2idx = word_vocab, pos_vocab

        self.data = self.make_tensor(data)

    def parse_line(self, path):
        print("parse line start...")
        hannanum = Hannanum()
        lines = [line.strip() for line in open(path, "r", encoding="utf-8")]
        data = []
        # self.max_sentence_length = 0
        for line in lines:
            sbj, obj, rel, src = line.split("\t")

            src = src.replace("<< _sbj_ >>", "<<_sbj_>>")
            src = src.replace("<< _obj_ >>", "<<_obj_>>")

            tokens = src.split()
            e1 = tokens.index("<<_sbj_>>")
            e2 = tokens.index("<<_obj_>>")

            src = src.replace("<<_sbj_>>", sbj)
            src = src.replace("<<_obj_>>", obj)

            # src = re.sub(r'\([^)]*\)', '', src)
            src = src.strip()

            # tokens = [p[0] + '/' + p[1] for p in hannanum.pos(src)]
            tokens = hannanum.morphs(src)
            print(tokens)
            # self.max_sentence_length = max(self.max_sentence_length, len(tokens))

            pos1, pos2 = self.get_relative_position(tokens, e1, e2)

            rel = self.labels[rel]

            tokens = tokens[:self.max_sentence_length] + \
                     ['UNK' for _ in range(self.max_sentence_length - len(tokens))]

            dat = {'sbj': sbj,
                   'obj': obj,
                   'rel': rel,
                   'src': src,
                   'tokens': tokens,
                   'pos1': pos1,
                   'pos2': pos2}
            data.append(dat)
        return data

    def make_vocab(self, data):
        print("make vocab")
        word2idx = {}
        pos2idx = {}

        for datum in data:
            for tok in datum['tokens']:
                if word2idx.get(tok) == None:
                    word2idx[tok] = len(word2idx)

            for pos in datum['pos1'] + datum['pos2']:
                if pos2idx.get(pos) == None:
                    pos2idx[pos] = len(pos2idx)

        # word2idx['UNK'] = len(word2idx)
        return word2idx, pos2idx

    def make_tensor(self, data):
        tensors = []
        for dat in data:
            tokens = torch.LongTensor([self.word2idx[tok] if tok in self.word2idx else self.word2idx['UNK'] for tok in dat['tokens']])
            pos1 = torch.LongTensor([self.pos2idx[pos] for pos in dat['pos1']])
            pos2 = torch.LongTensor([self.pos2idx[pos] for pos in dat['pos2']])
            t = {
                'tokens': tokens,
                'pos1': pos1,
                'pos2': pos2,
                'rel': dat['rel']
            }
            tensors.append(t)
        return tensors

    def get_relative_position(self, tokens, e1, e2):
        # Position data
        pos1 = []
        pos2 = []

        for word_idx in range(len(tokens)):
            pos1.append(str(self.max_sentence_length - 1 + word_idx - e1))
            pos2.append(str(self.max_sentence_length - 1 + word_idx - e2))

        for _ in range(self.max_sentence_length - len(tokens)):
            pos1.append(str(999))
            pos2.append(str(999))

        return pos1, pos2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
