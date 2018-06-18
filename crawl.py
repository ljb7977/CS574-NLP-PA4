import re

from SPARQLWrapper import SPARQLWrapper, JSON


def sendQuery(item):
    prefix = 'prefix dbpedia: <http://ko.dbpedia.org/resource/> prefix dbpedia-owl: <http://dbpedia.org/ontology/> select ?abstract where {  <http://ko.dbpedia.org/resource/'
    suffix = '> dbpedia-owl:wikiPageRedirects*/dbpedia-owl:abstract ?abstract .}'
    query = prefix + item + suffix

    sparql = SPARQLWrapper("http://ko.dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    try:
        result = sparql.query().convert()['results']['bindings'][0]['abstract']['value']
        return str(result)
    except:
        return None


if __name__ == "__main__":
    datapath = "data/triples.nt"
    lines = [line.strip() for line in open(datapath, "r", encoding="utf-8")]

    with open("new_data.tsv", "w", encoding="utf-8") as output:
        for i, line in enumerate(lines):
            sbj, rel, obj, _ = line.split("\t")

            dat = sendQuery(sbj)

            sbj2 = re.sub(r'\(.*\)', "", sbj.replace("_", " ")).strip()
            obj2 = re.sub(r'\(.*\)', "", obj.replace("_", " ")).strip()

            print(i, "th entry : ", sbj2, rel, obj2)

            if dat is None:
                continue

            n = 1  # run at least once
            while n:
                dat, n = re.subn(r'\([^()]*\)', '', dat)  # remove non-nested/flat balanced parts

            sentences = [sentence+u"다." for sentence in dat.split(u"다.")[:-1]]
            for sentence in sentences:
                # print(sentence)
                sentence = sentence.strip()

                print(sentence)
                if sbj2 in sentence:
                    result = sentence.replace(sbj2, " << _sbj_ >> ", 1)
                    if obj2 in result:
                        result = result.replace(obj2, " << _obj_ >> ", 1)
                        print("\t".join((sbj, obj, rel, result)), file=output)