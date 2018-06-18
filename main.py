import train
import numpy

def analysis(name, result):
    print("Analysis start: ", name)
    with open("analysis"+name+".txt", "w") as output:
        print("model: ", name, file=output)
        print("threshold precision recall", file=output)

        t = [[]]*6
        for threshold in numpy.arange(0.1, 1, 0.1):
            precision, recall = PR(threshold, result)

            for pre, rec in zip(precision, recall):
                print(round(threshold, 3), round(pre, 3), round(rec, 3), file=output, end=" ")
            print(file=output)

    print("Analysis Done")
    return


def PR(threshold, result):
    predicted = [0, 0, 0, 0, 0, 0]
    gold = [0, 0, 0, 0, 0, 0]
    correct = [0, 0, 0, 0, 0, 0]

    for r in result:
        label, tensor = r
        for i, elem in enumerate(tensor):
            if elem > threshold:
                predicted[i] += 1
                gold[label] += 1
                if label == i:
                    correct[i] += 1

    precision = [0, 0, 0, 0, 0, 0]
    recall = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        if predicted[i] != 0:
            precision[i] = correct[i]/predicted[i]
        if gold[i] != 0:
            recall[i] = correct[i]/gold[i]

    return precision, recall


if __name__ == "__main__":
    trainer = train.Trainer()
    # trainer.train()
    for name in ["0_82.6.model", "1_81.5.model", "2_79.8.model"]:
        result = trainer.test(load=True, model_name=name)
        analysis(name, result)
