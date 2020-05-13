# coding: utf-8

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "..")

from text_tree.tree import Tree


def test_tree():
    texts = [
        "出栏一头猪亏损300元，究竟谁能笑到最后！",
        "以前很火的巴铁为何现在只字不提？",
        "出栏一头牛亏损300元，究竟谁能笑到最后！",
    ]

    tree = Tree(label="商业", root="亏损", left_has=['猪', '牛'])
    assert tree.match(texts[0])
    assert not tree.match(texts[1])
    assert tree.match(texts[2])

    tree = Tree(label="商业", root="亏损", left_has=['.{10,20}猪', '牛'], use_re=True)
    assert not tree.match(texts[0])
    assert not tree.match(texts[1])
    assert tree.match(texts[2])

    tree = Tree(label="商业", root=["亏损", "巴铁"], has=['猪', "现在"], use_re=False)
    assert tree.match(texts[0])
    assert tree.match(texts[1])
    assert not tree.match(texts[2])
