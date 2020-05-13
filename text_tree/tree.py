# coding: utf-8
import re
from collections import Counter


class Tree(object):
    """特征组合 - 以树的形式展现"""
    def __init__(self, label, root, use_re=False,
                 has=None, without=None,
                 left_has=None, right_has=None,
                 left_without=None, right_without=None):
        self.label = label
        self.root = root
        self.has = has
        self.without = without
        self.left_has = left_has
        self.right_has = right_has
        self.left_without = left_without
        self.right_without = right_without
        self.use_re = use_re

    def __repr__(self):
        r = "use_re: " + str(self.use_re) + "\n"
        r += "root: " + str(self.root) + "\n"
        r += "    " + "left_has: " + str(self.left_has) + "\n"
        r += "    " + "right_has: " + str(self.right_has) + "\n"
        r += "    " + "left_without: " + str(self.left_without) + "\n"
        r += "    " + "right_without: " + str(self.right_without) + "\n"
        return "<%s>" % r.strip("\n")

    def _features_in(self, features, text):
        use_re = self.use_re
        if use_re:
            c = [x for x in features if re.search(x, text)]
        else:
            c = [x for x in features if x in text]

        if c:
            return True
        else:
            return False

    def match(self, text):
        """使用 tree 对 text 进行匹配"""
        root = self.root
        if isinstance(root, str):
            # root 是 str 的情况下，才能将 text 拆分成 left、right
            if self.use_re:
                if not re.search(self.root, text):
                    return False
                left, right = re.split(self.root, text, 1)
            else:
                if self.root not in text:
                    return False
                left, right = text.split(self.root, 1)

            if self.has and len(self.has) > 0 and not self._features_in(self.has, text):
                return False

            if self.left_has and len(self.left_has) > 0 and not self._features_in(self.left_has, left):
                return False

            if self.right_has and len(self.right_has) > 0 and not self._features_in(self.right_has, right):
                return False

            if self.without and len(self.without) > 0 and self._features_in(self.without, text):
                return False

            if self.left_without and len(self.left_without) > 0 and self._features_in(self.left_without, left):
                return False

            if self.right_without and len(self.right_without) > 0 and self._features_in(self.right_without, right):
                return False

            return True
        elif isinstance(root, list):
            if not self._features_in(root, text):
                return False

            # 对 has 进行判断
            if self.has and len(self.has) > 0 and not self._features_in(self.has, text):
                return False

            # 对 without 进行判断
            if self.without and len(self.without) > 0 and self._features_in(self.without, text):
                return False
            return True
        else:
            raise ValueError('`root` must be instance of str or list')


class Forest(object):
    def __init__(self):
        self.trees = []

    def add_tree(self, tree):
        self.trees.append(tree)

    def predict(self, text, return_at_first=True):
        if return_at_first:
            for tree in self.trees:
                if tree.match(text):
                    return tree.label
            return None
        else:
            pred_labels = []
            for tree in self.trees:
                if tree.match(text):
                    pred_labels.append(tree.label)

            if pred_labels:
                c = Counter(pred_labels)
                label = c.most_common(1)[0]
                return label
            else:
                return None


