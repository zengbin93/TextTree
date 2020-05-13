# TextTree

文本分类树：自动化文本分类规则生成

## 基本思想

基于统计的方法，查找在某一类文本中出现次数明显大于其他类文本的组合特征。
在这个库中，组合特征主要以`tree`的形式程序，由此取名`TextTree`。

`Tree` 至少包含 label 和 root 信息，Tree 所覆盖的样本，必然包含 root 特征。
left_has, right_has, left_without, right_without 是四个可选的叶子节点，
叶子节点用于提高 Tree 针对 label 的分类准确率， 
每个节点都是 list 类型的数据，可以为空。


可调参数：

* max_feature_labels - 单一特征的最大覆盖类别数
* min_feature_distinct - 单一特征针对某一类别的最小类别辨识度


假设数据集 `D = [{text_1, label_1}, {text_2, label_2}, ... , {text_m, label_n}]`
，其中`label_set`的大小为 `n`，`text_set` 的大小为 `m`，
其中 `label_1` 表示第一类文本，查找过程如下：

1. 遍历 text_set，抽取全部语料的 n-gram 特征，逐一统计特征在各类别中出现情况，
    如 覆盖样本数、覆盖类别数、类别辨识度
  

