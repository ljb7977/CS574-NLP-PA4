import torch
from konlpy.tag import Hannanum

def load_data_and_labels(path):
    data = []
    lines = [line.strip() for line in open(path, "r", encoding="utf-8")]

    hannanum = Hannanum()

    word2idx = {}
    pos2idx = {}

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
        for tok in tokens:
            if word2idx.get(tok) == None:
                word2idx[tok]=len(word2idx)
        print(e1, e2)

        labels = {
            'country': 1, 'team': 2,
            'starring': 3, 'director': 4,
            'child': 5, 'successor': 6
        }

        rel = labels[rel]
        print('rel: ', rel)

        pos1, pos2 = get_relative_position(tokens, e1, e2)
        for pos in pos1+pos2:
            if pos2idx.get(pos) == None:
                pos2idx[pos] = len(pos2idx)

        dat = {'sbj': sbj,
               'obj': obj,
               'rel': rel,
               'src': src,
               'tokens': tokens,
               'pos1': pos1,
               'pos2': pos2}

        data.append(dat)

    for dat in data:
        tokens = dat['tokens']
        pos1 = dat['pos1']
        pos2 = dat['pos2']

        tokens = torch.LongTensor([word2idx[tok] for tok in tokens])
        pos1 = torch.LongTensor([pos2idx[pos] for pos in pos1])
        pos2 = torch.LongTensor([pos2idx[pos] for pos in pos2])

        dat['tokens'] = tokens
        dat['pos1'] = pos1
        dat['pos2'] = pos2

    return data, word2idx, pos2idx

def get_relative_position(tokens, e1, e2, max_sentence_length=111):
    # Position data
    pos1 = []
    pos2 = []

    for word_idx in range(len(tokens)):
        pos1.append(str(max_sentence_length - 1 + word_idx - e1))
        pos2.append(str(max_sentence_length - 1 + word_idx - e2))

    for _ in range(max_sentence_length - len(tokens)):
        pos1.append(str(999))
        pos2.append(str(999))

    return pos1, pos2
