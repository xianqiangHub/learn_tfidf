#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    corpus = ["我 来到 北京 清华大学",
              "他 来到 了 网易 杭研 大厦",
              "小明 硕士 毕业 与 中国 科学院",
              "我 爱 北京 天安门"]
    vectorizer = TfidfVectorizer()

    # 直接是特征权重
    transform = vectorizer.fit_transform(corpus)

    # 词袋
    words = vectorizer.get_feature_names()

    print(words)
    print(transform.toarray())