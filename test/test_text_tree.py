# coding: utf-8
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "..")

from text_tree.text_tree import ngram, extract_features, TextTree

lines = open(r"C:\data\nlp\tnews\train.json", encoding='utf-8').readlines()
data = []
for line in lines:
    line = eval(line.strip("\n"))
    row = {
        'text': line['sentence'],
        "label": line['label']
    }
    data.append(row)

# data = data[:1000]


def test_ngram():
    x = "12345"
    grams = ngram(x, n=3)
    print(grams)
    assert grams == ["1", "2", "3", "4", "5", "12", "23", "34", "45", "123", "234", "345"]


def test_extract_ngram_features():
    features = extract_features(data, n=4)
    assert isinstance(features, dict)


def test_text_tree():
    tt = TextTree(data, n=4)
    tt.fit_one_label('104')


