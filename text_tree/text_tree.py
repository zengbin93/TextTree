# coding: utf-8
from tqdm import tqdm
from functools import lru_cache
from collections import Counter
import math
from .tree import Tree


def char_tokenizer(text):
    return list(text)


def space_tokenizer(text):
    return text.split(" ")


@lru_cache(maxsize=65536 * 2, typed=False)
def ngram(text, n=4, tokenizer=char_tokenizer):
    tokens = tokenizer(text)
    feature = []
    for i in range(1, n + 1):
        feature.extend(["".join(tokens[j:j + i]) for j in range(len(tokens) - i + 1)])
    return feature


class TextTree:
    """
    算法描述：

    输入数据：data

    可调参数：
        n - ngram 参数
        tokenizer - 分词器
        min_tree_distinct - 单棵树的最小类别区分度


    """

    def __init__(self, data, n=4, tokenizer=char_tokenizer,
                 min_tree_distinct=0.9, min_root_distinct=0.6):
        self.data = data
        self.n = n
        self.tokenizer = tokenizer
        self.min_tree_distinct = min_tree_distinct
        self.min_root_distinct = min_root_distinct

        self.labels = list(set([x['label'] for x in data]))
        self.idf = self.features_idf(data)
        self.features = self.extract_features(data)
        self.distinct = dict()
        for k, v in self.features.items():
            t = sum(v.values())
            v_distinct = {x: y / t for x, y in v.items()}
            self.distinct[k] = v_distinct
        self.__label_features()

    def extract_features(self, data):
        """抽取文档特征"""
        n = self.n
        tokenizer = self.tokenizer
        n_keys = 100

        features = dict()
        for row in data:
            label = row['label']
            text = row['text']
            grams = ngram(text, n, tokenizer)
            c = Counter(grams)
            grams_tfidf = [(k, v * self.idf.get(k, 0)) for k, v in c.items()]
            grams_tfidf = sorted(grams_tfidf, key=lambda x: x[1], reverse=True)[:n_keys]

            for gram, _ in grams_tfidf:
                feature = features.get(gram, {})
                num = feature.get(label, 0)
                num += 1
                feature[label] = num
                features[gram] = feature

        return features

    def features_idf(self, data):
        """文本的 idf 特征"""
        n = self.n
        tokenizer = self.tokenizer
        corpus = [x['text'] for x in data]
        vocab = dict()
        for i, text in enumerate(corpus):
            tokens = ngram(text, n, tokenizer)
            for token in set(tokens):
                doc_index = vocab.get(token, [])
                doc_index.append(i)
                vocab[token] = doc_index

        idf = dict()
        total_doc = len(corpus)
        for token, doc_index in tqdm(vocab.items(), desc="features_idf"):
            num = len(doc_index)
            idf_ = math.log(total_doc / (num + 1))
            idf[token] = idf_
        return idf

    def __label_features(self):
        """构造每一个类别的特征"""
        label_features = dict()
        labels = self.labels
        for label in labels:
            data_l = [x for x in self.data if x['label'] == label]
            f = self.extract_features(data_l)
            f1 = [(x, y) for x, v in f.items() for _, y in v.items()]
            # 按 feature 覆盖的样本量从大到小排序
            label_features[label] = sorted(f1, key=lambda x: x[1], reverse=True)

        self.label_features = label_features

    def _generate_one_tree(self, root, label):
        """给定 root 和 label 生成一颗分类规则树"""
        dr = [x for x in self.data if root in x['text']]
        dl = [x['text'] for x in dr if x['label'] == label]
        dl_text = "".join(dl)
        tree = Tree(root=root, label=label)
        p = tree.evaluate(dr)
        print("tree v0:", tree)

        if p >= self.min_tree_distinct:
            return tree
        else:
            left = []
            right = []
            for row in dr:
                l_, r_ = row['text'].split(root, 1)
                left.append({"text": l_, "label": row['label']})
                right.append({"text": r_, "label": row['label']})
            left_f = self.extract_features(left)
            right_f = self.extract_features(right)
            # print(left_f, right_f)

            max_n = 100
            # left without
            f1 = [(x, y) for x, v in left_f.items() for l1, y in v.items() if l1 != label and x not in dl_text]
            f1 = sorted(f1, key=lambda x: x[1], reverse=True)
            left_without = [x[0] for x in f1[:max_n]]
            tree = Tree(root=root, label=label, left_without=left_without)
            print("tree v1:", tree)
            p = tree.evaluate(dr)
            if p >= self.min_tree_distinct:
                return tree

            # right without
            f1 = [(x, y) for x, v in right_f.items() for l1, y in v.items() if l1 != label and x not in dl_text]
            f1 = sorted(f1, key=lambda x: x[1], reverse=True)
            right_without = [x[0] for x in f1[:max_n]]
            tree = Tree(root=root, label=label, left_without=left_without, right_without=right_without)
            p = tree.evaluate(dr)
            print("tree v2:", tree)
            if p >= self.min_tree_distinct:
                return tree

            # left has
            f1 = [(x, y) for x, v in left_f.items() for l1, y in v.items() if l1 == label and x in dl_text]
            f1 = sorted(f1, key=lambda x: x[1], reverse=True)
            left_has = [x[0] for x in f1[:max_n]]
            tree = Tree(root=root, label=label, left_without=left_without,
                        right_without=right_without, left_has=left_has)
            p = tree.evaluate(dr)
            print("tree v3:", tree)
            if p >= self.min_tree_distinct:
                return tree

            # right has
            f1 = [(x, y) for x, v in right_f.items() for l1, y in v.items() if l1 == label and x in dl_text]
            f1 = sorted(f1, key=lambda x: x[1], reverse=True)
            right_has = [x[0] for x in f1[:max_n]]
            tree = Tree(root=root, label=label, left_without=left_without, right_without=right_without,
                        left_has=left_has, right_has=right_has)
            p = tree.evaluate(dr)
            print("tree v4:", tree)
            if p >= self.min_tree_distinct:
                return tree

            # 如果最后没有能够生成满足条件的树，返回 None
            return None

    def fit_one_label(self, label):
        """针对某一类别的规则生成

        1. 使用该类别下区分度为 1 的特征构造精确匹配分类树；
        2. 获取该类别下区分度大于 min_root_distinct 的特征作为 root 依次生成分类树；
        3.

        :param label: str or int
            类别标签
        :return:
        """
        lf = self.label_features[label]
        trees = []
        # potential_root = [x for x, _ in lf if 0.9 > self.distinct[x][label] >= self.min_root_distinct]
        potential_root = [x for x, num in lf if num > 10]
        for root in tqdm(potential_root, desc=str(label)):
            tree = self._generate_one_tree(root, label)
            if isinstance(tree, Tree):
                trees.append(tree)
        return trees
