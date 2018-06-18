import random
if __name__ == "__main__":
    new_data = [line.strip() for line in open("data/new_data.tsv", "r", encoding="utf-8")]
    new_test = [line.strip() for line in open("data/new_test.tsv", "r", encoding="utf-8")]
    new_train = [line.strip() for line in open("data/new_train.tsv", "r", encoding="utf-8")]

    mass_data = new_data+new_test+new_train
    random.shuffle(mass_data)

    print(len(mass_data))

    idx = len(mass_data) * 9 // 10
    train_set = mass_data[:idx]
    test_set = mass_data[idx:]

    with open("data/large_train.tsv", "w", encoding="utf-8") as output:
        for i in train_set:
            print(i, file=output)

    with open("data/large_test.tsv", "w", encoding="utf-8") as output:
        for i in test_set:
            print(i, file=output)
