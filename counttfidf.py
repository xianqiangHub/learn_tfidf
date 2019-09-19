#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

if __name__ == '__main__':
    corpus = ["我 来到 北京 清华大学",
              "他 来到 了 网易 杭研 大厦",
              "小明 硕士 毕业 与 中国 科学院",
              "我 爱 北京 天安门"]
    vectorizer = CountVectorizer(analyzer='word')
    # 拿到词频矩阵
    word_matrix = vectorizer.fit_transform(corpus)
    # 词袋
    words = vectorizer.get_feature_names()
    # print(words)
    # print(word_matrix.toarray())
    transformer = TfidfTransformer()
    weight = transformer.fit_transform(word_matrix)
    toarray = weight.toarray()
    for i in range(len(toarray)):
        print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
        for j in range(len(words)):
            print(words[j], toarray[i][j])

    print(weight.toarray())
