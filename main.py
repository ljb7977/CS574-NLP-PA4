from konlpy.tag import hannanum

if __name__ == "__main__":
    data = []
    hannanum = Hannanum()
    with open("data/ds_train.tsv", "r", encoding="utf-8") as file:
        while True:
            line = file.readline().rstrip()
            if not line:
                break
            sub, obj, rel, src = line.split("\t")

            tokens = [p[0]+'/'+p[1] for p in hannanum.pos(src)]
            print(tokens)

            dat = {'sub': sub,
                   'obj': obj,
                   'rel': rel,
                   'src': src}
            data.append(dat)
            print(dat)