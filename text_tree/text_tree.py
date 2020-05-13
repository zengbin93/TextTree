# coding: utf-8
from functools import lru_cache
from .tree import Tree


def char_tokenizer(text):
    return list(text)


def space_tokenizer(text):
    return text.split(" ")


@lru_cache(maxsize=65536*2, typed=False)
def ngram(text, n=4, tokenizer=char_tokenizer):
    tokens = tokenizer(text)
    feature = []
    for i in range(1, n+1):
        feature.extend(["".join(tokens[j:j + i]) for j in range(len(tokens)-i+1)])
    return feature


def extract_features(data, n=4, tokenizer=char_tokenizer):
    """抽取文档特征

    :param tokenizer: function
        分词器
    :param data: list of dict
        文本集合
    :param n: int
        n-gram 参数
    :return: OrderedDict
        {"feature": {"label1": num1, "label2": num2}}
    """
    features = dict()
    for row in data:
        label = row['label']
        text = row['text']
        grams = ngram(text, n, tokenizer)

        for gram in set(grams):
            feature = features.get(gram, {})
            num = feature.get(label, 0)
            num += 1
            feature[label] = num
            features[gram] = feature

    return features


class TextTree:
    def __init__(self, data, n=4, tokenizer=char_tokenizer,
                 min_tree_distinct=0.9):
        self.data = data
        self.labels = list(set([x['label'] for x in data]))
        # self.features = extract_features(data, n, tokenizer)

        self.n = n
        self.tokenizer = tokenizer
        self.min_tree_distinct = min_tree_distinct
        self.__label_features()

    def __label_features(self):
        """构造每一个类别的特征"""
        label_features = dict()
        labels = self.labels
        for label in labels:
            data_l = [x for x in self.data if x['label'] == label]
            f = extract_features(data_l, self.n, self.tokenizer)
            f1 = [(x, y) for x, v in f.items() for _, y in v.items()]
            # 按 feature 覆盖的样本量从大到小排序
            label_features[label] = sorted(f1, key=lambda x: x[1], reverse=True)

        self.label_features = label_features

    @staticmethod
    def _tree_distinct(tree, data):
        matched = [x for x in data if tree.match(x['text'])]
        true_matched = [x for x in matched if x['label'] == tree.label]
        return len(true_matched) / (len(matched) + 0.0001)

    def _generate_one_tree(self, root, label):
        """给定 root 和 label 生成一颗分类规则树"""
        print("\n\n", root, label, "=" * 100)
        dr = [x for x in self.data if root in x['text']]
        tree = Tree(root=root, label=label)
        p = self._tree_distinct(tree, dr)
        print("root distinct: ", p)
        if p >= self.min_tree_distinct:
            return tree
        else:
            left = []
            right = []
            for row in dr:
                l, r = row['text'].split(root, 1)
                left.append({"text": l, "label": label})
                right.append({"text": r, "label": label})
            left_f = extract_features(left, self.n, self.tokenizer)
            right_f = extract_features(right, self.n, self.tokenizer)

            # left without
            f1 = [(x, y) for x, v in left_f.items() for l, y in v.items() if l != label]
            f1 = sorted(f1, key=lambda x: x[1], reverse=True)
            left_without = [x[0] for x in f1[:20]]

            # left has
            f1 = [(x, y) for x, v in left_f.items() for l, y in v.items() if l == label]
            f1 = sorted(f1, key=lambda x: x[1], reverse=True)
            left_has = [x[0] for x in f1[:20]]

            # right without
            f1 = [(x, y) for x, v in right_f.items() for l, y in v.items() if l != label]
            f1 = sorted(f1, key=lambda x: x[1], reverse=True)
            right_without = [x[0] for x in f1[:20]]

            # right has
            f1 = [(x, y) for x, v in right_f.items() for l, y in v.items() if l == label]
            f1 = sorted(f1, key=lambda x: x[1], reverse=True)
            right_has = [x[0] for x in f1[:20]]

            tree_list = [
                Tree(root=root, label=label, left_without=left_without),
                Tree(root=root, label=label, left_without=left_without, right_without=right_without),
                Tree(root=root, label=label, left_without=left_without, right_without=right_without, left_has=left_has),
                Tree(root=root, label=label, left_without=left_without, right_without=right_without,
                     left_has=left_has, right_has=right_has)
            ]
            for tree in tree_list:
                p = self._tree_distinct(tree, dr)
                print(tree, "\ndistinct: ", p)
                if p >= self.min_tree_distinct:
                    return tree

            # 如果最后没有能够生成满足条件的树，返回 None
            return None

    def fit_one_label(self, label):
        features = self.label_features[label]
        trees = []
        for root, num in features:
            print(root)
            tree = self._generate_one_tree(root, label)
            if isinstance(tree, Tree):
                trees.append(tree)
        return trees




