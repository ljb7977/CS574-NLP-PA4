import requests

page_prefix = "https://ko.wikipedia.org/api/rest_v1/page/summary/"


def get_wiki(sbj):
    try:
        res = requests.get(page_prefix+sbj)
        content = res.json()['extract']
        return content.replace("\n", "")
    except Exception:
        return None


def make_simply(word):
    word = word.replace("_", " ").strip()
    left = word.find("(")
    if left == -1:
        return word
    right = word.find(")")
    if right == -1:
        return word

    return word[:left] + word[right + 1:]


def reduce_content(content):
    sbj_loc = content.find("<< _sbj_ >>")
    obj_loc = content.find("<< _obj_ >>")
    start_p = max(sbj_loc, obj_loc) + 10

    for i in range(start_p, len(content)):
        if content[i] == ".":
            return content[:i+1]
    return content


def try_to_changing(triple_format):
    parsed = triple_format.split("\t")
    sbj = parsed[0]
    relation = parsed[1]
    obj = parsed[2]
    content = get_wiki(sbj)
    if content is None:
        return triple_format
    sbj_simple = make_simply(sbj)
    obj_simple = make_simply(obj)

    if sbj_simple in content and obj_simple in content:
        changed_content = reduce_content(content.replace(sbj_simple, "<< _sbj_ >>").replace(obj_simple, "<< _obj_ >>"))
        return sbj + "\t" + obj + "\t" + relation + "\t" + changed_content + "\n"
    else:
        return triple_format


def main():
    f1 = open('./data/triples.nt', 'r', encoding='utf-8')
    f2 = open('./data/triples_new.nt', 'w', encoding='utf-8')
    i = 1
    while True:
        print(i)
        i += 1
        origin = f1.readline()
        if origin == '':
            f2.write(origin)
            break
        f2.write(try_to_changing(origin))
    f1.close()
    f2.close()


if __name__ == "__main__":
    main()
