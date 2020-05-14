# coding: utf-8
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "..")
from tqdm import tqdm
from text_tree.text_tree import ngram, TextTree
from text_tree.tree import Forest


def read_data(file):
    lines = open(file, encoding='utf-8').readlines()
    data = []
    for line in lines:
        line = eval(line.strip("\n"))
        row = {
            'text': line['sentence'],
            "label": line['label']
        }
        data.append(row)
    return data


train = read_data(r"C:\data\nlp\tnews\train.json")
dev = read_data(r"C:\data\nlp\tnews\dev.json")


def test_ngram():
    x = "12345"
    grams = ngram(x, n=3)
    print(grams)
    assert grams == ["1", "2", "3", "4", "5", "12", "23", "34", "45", "123", "234", "345"]


def test_text_tree():
    tt = TextTree(train, n=4, min_tree_distinct=0.9, min_root_distinct=0.7)
    label = "106"
    trees = tt.fit_one_label(label)

    forest = Forest(trees)
    pred = forest.evaluate(dev)
    dl = [x for x in pred if x['label'] == label]
    pl = [x for x in pred if x['pred'] == label]
    tl = [x for x in pl if x['pred'] == x['label']]
