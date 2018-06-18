import re

if __name__ == "__main__":
    datapath = "data/gold_test.tsv"
    lines = [line.strip() for line in open(datapath, "r", encoding="utf-8")]

    with open("new_test.tsv", "w", encoding="utf-8") as output:
        for line in lines:
            sbj, obj, rel, src = line.split("\t")

            sbj2 = re.sub(r'\(.*\)', "", sbj.replace("_", " ")).strip()
            obj2 = re.sub(r'\(.*\)', "", obj.replace("_", " ")).strip()

            n = 1  # run at least once
            while n:
                src, n = re.subn(r'\([^()]*\)', '', src)  # remove non-nested/flat balanced parts

            if "<< _sbj_ >>" in src and "<< _obj_ >>" in src:
                print("\t".join((sbj, obj, rel, src)), file=output)