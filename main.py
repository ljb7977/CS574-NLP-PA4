import train

if __name__ == "__main__":
    mode = 1
    if mode == 1:
        train_path = "data/new_train.tsv"
        test_path = "data/new_test.tsv"
    elif mode == 2:
        train_path = "data/large_train.tsv"
        test_path = "data/large_test.tsv"

    trainer = train.Trainer(train_path=train_path, test_path=test_path)
    # trainer.train(mode=mode)
    for i in range(5):
        trainer.test(epoch=i, load=True, model_path="model/1/epoch_"+str(i)+".model")
    # trainer.test(load=True, model_path="model/2/0_81.64_model.model")
