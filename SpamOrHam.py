import os.path
from builtins import range

import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle

okt = Okt()
if not os.path.isfile('new_blog_spam5.txt'):
    f = open("new_blog_spam5.txt", 'w', encoding='UTF8')
    for line in open('blog_spam.txt', encoding='UTF8'):
        temp = line.split('\t')
        new_text = ''
        for i in range(4, len(temp)):
            new_text += ' ' + ' '.join([t[0] for t in okt.pos(temp[i], norm=True, stem=True)])

        new_line = '\t'.join(temp[:3]) + '\t' + new_text
        f.write(new_line)
    f.close()

df = pd.read_csv("new_blog_spam.txt", sep='\t', names=["type", "label", "docid", "docs"])

df['length'] = df['docs'].apply(len)
print(df.groupby("label").describe())

# import matplotlib.pyplot as plt
# #
# # plt.hist(df['length'])
# # plt.show()
#
# print(df['length'].describe())
#
# print(df[df['length'] == 1485]['docs'].iloc[0])

train_set = df[df['type'] == 'tr']
train_set = shuffle(train_set)
test_set = df[df['type'] == 'ts']
test_set = shuffle(test_set)

### Bag of Words 및 나이브 베이즈 활용 분류
cv = CountVectorizer()


bow_transformer = cv.fit(df['docs'])
docs_bow = bow_transformer.transform(train_set['docs'])

spam_detect_model = MultinomialNB(alpha=0.0001).fit(docs_bow, train_set['label'])
all_predictions = spam_detect_model.predict(docs_bow)
print(classification_report(train_set['label'], all_predictions))
print('\n')
m_confusion_test = confusion_matrix(train_set['label'], all_predictions)
print(pd.DataFrame(data=m_confusion_test))
print(spam_detect_model.score(docs_bow, train_set['label']))

print(spam_detect_model.score(bow_transformer.transform(test_set['docs']), test_set['label']))

##tf-idf 활용
tfidf_transformer = TfidfTransformer().fit(docs_bow)
docs_tfidf = tfidf_transformer.transform(docs_bow)

tfidf_detect_model = MultinomialNB(alpha=0.0001).fit(docs_tfidf, train_set['label'])
tfidf_predictions = tfidf_detect_model.predict(docs_tfidf)

print(classification_report(train_set['label'], tfidf_predictions))
print('\n')
m_confusion_test = confusion_matrix(train_set['label'], tfidf_predictions)
print(pd.DataFrame(data=m_confusion_test))

def binary_classifier(df):
    train_set
    cv = CountVectorizer()
    bow_transformer = cv.fit(train_set['docs'])
    docs_bow = bow_transformer.transform(train_set['docs'])
    docs_test = bow_transformer.transform(df['docs'])

    ### 정상 비정상 분류
    normal_label = np.where(train_set['label'] == '정상', '정상', '비정상')
    normal_detect_model = MultinomialNB(alpha=0).fit(docs_bow, normal_label)
    normal_predictions = normal_detect_model.predict(docs_test)
    normal_selector = []

    for i in range(len(normal_predictions)):
        if normal_predictions[i] == '비정상':
            normal_selector.append(i)

    next_set = df.iloc[normal_selector]

    ### 비정상데이터 Bag of words
    spam_train_set = df[df['label'] != '정상']
    spam_bow_transformer = cv.fit(spam_train_set['docs'])
    spam_docs_bow = spam_bow_transformer.transform(spam_train_set['docs'])
    spam_docs_test = spam_bow_transformer.transform(next_set['docs'])

    ### 대량 비대량 분류
    large_label = np.where(spam_train_set['label'] == '대량', '대량', '비대량')
    large_detect_model = MultinomialNB(alpha=0).fit(spam_docs_bow, large_label)
    large_predictions = large_detect_model.predict(spam_docs_test)

    large_selector = []
    for i in range(len(large_predictions)):
        if large_predictions[i] == '비대량':
            large_selector.append(i)

    next_set = next_set.iloc[large_selector]

    i = 0
    j = 0
    while i < len(normal_predictions):
        if normal_predictions[i] == '비정상':
            normal_predictions[i] = large_predictions[j]
            j += 1
        i += 1

    ### 비대량데이터 Bag of words
    spam_train_set_2 = spam_train_set[spam_train_set['label'] != '대량']
    spam_bow_transformer_2 = cv.fit(spam_train_set_2['docs'])
    spam_docs_bow_2 = spam_bow_transformer_2.transform(spam_train_set_2['docs'])
    spam_docs_test_2 = spam_bow_transformer_2.transform(next_set['docs'])

    ### 불법 합법 분류
    illegal_label = np.where(spam_train_set_2['label'] == '불법', '불법', '합법')
    illegal_detect_model = MultinomialNB(alpha=0).fit(spam_docs_bow_2, illegal_label)
    illegal_predictions = illegal_detect_model.predict(spam_docs_test_2)

    illegal_selector = []
    for i in range(len(illegal_predictions)):
        if illegal_predictions[i] == '합법':
            illegal_selector.append(i)

    next_set = next_set.iloc[illegal_selector]

    i = 0
    j = 0
    while i < len(normal_predictions):
        if normal_predictions[i] == '비대량':
            normal_predictions[i] = illegal_predictions[j]
            j += 1
        i += 1

    ### 비불법데이터 Bag of words
    spam_train_set_3 = spam_train_set_2[spam_train_set_2['label'] != '불법']
    spam_bow_transformer_3 = cv.fit(spam_train_set_3['docs'])
    spam_docs_bow_3 = spam_bow_transformer_3.transform(spam_train_set_3['docs'])
    spam_docs_test_3 = spam_bow_transformer_3.transform(next_set['docs'])

    ### 도배 비도배 분류
    duplicate_label = np.where(spam_train_set_3['label'] == '도배', '도배', '비도배')
    duplicate_detect_model = MultinomialNB(alpha=0).fit(spam_docs_bow_3, duplicate_label)
    duplicate_predictions = duplicate_detect_model.predict(spam_docs_test_3)

    duplicate_selector = []
    for i in range(len(duplicate_predictions)):
        if duplicate_predictions[i] == '비도배':
            duplicate_selector.append(i)

    next_set = next_set.iloc[duplicate_selector]

    i = 0
    j = 0
    while i < len(normal_predictions):
        if normal_predictions[i] == '합법':
            normal_predictions[i] = duplicate_predictions[j]
            j += 1
        i += 1

    ### 비도배데이터 Bag of words
    spam_train_set_4 = spam_train_set_3[spam_train_set_3['label'] != '도배']
    spam_bow_transformer_4 = cv.fit(spam_train_set_4['docs'])
    spam_docs_bow_4 = spam_bow_transformer_4.transform(spam_train_set_4['docs'])
    spam_docs_test_4 = spam_bow_transformer_4.transform(next_set['docs'])

    ### 홍보 여부 분류
    ad_label = np.where(spam_train_set_4['label'] == '홍보', '홍보', '비홍보')
    ad_detect_model = MultinomialNB(alpha=0).fit(spam_docs_bow_4, ad_label)
    ad_predictions = ad_detect_model.predict(spam_docs_test_4)

    ad_selector = []
    for i in range(len(ad_predictions)):
        if ad_predictions[i] == '비홍보':
            ad_selector.append(i)

    next_set = next_set.iloc[ad_selector]

    i = 0
    j = 0
    while i < len(normal_predictions):
        if normal_predictions[i] == '비도배':
            normal_predictions[i] = ad_predictions[j]
            j += 1
        i += 1

    ### 비홍보데이터 Bag of words
    spam_train_set_5 = spam_train_set_4[spam_train_set_4['label'] != '홍보']
    spam_bow_transformer_5 = cv.fit(spam_train_set_5['docs'])
    spam_docs_bow_5 = spam_bow_transformer_5.transform(spam_train_set_5['docs'])
    spam_docs_test_5 = spam_bow_transformer_5.transform(next_set['docs'])

    ### 도박 비도박 분류
    gamble_label = np.where(spam_train_set_5['label'] == '도박', '도박', '비도박')
    gamble_detect_model = MultinomialNB(alpha=0).fit(spam_docs_bow_5, gamble_label)
    gamble_predictions = gamble_detect_model.predict(spam_docs_test_5)

    gamble_selector = []
    for i in range(len(gamble_predictions)):
        if gamble_predictions[i] == '비도박':
            gamble_selector.append(i)

    next_set = next_set.iloc[gamble_selector]

    i = 0
    j = 0
    while i < len(normal_predictions):
        if normal_predictions[i] == '비홍보':
            normal_predictions[i] = gamble_predictions[j]
            j += 1
        i += 1
    print(i)

    ### 비도박데이터 Bag of words
    spam_train_set_6 = spam_train_set_5[spam_train_set_5['label'] != '도박']
    spam_bow_transformer_6 = cv.fit(spam_train_set_6['docs'])
    spam_docs_bow_6 = spam_bow_transformer_6.transform(spam_train_set_6['docs'])
    spam_docs_test_6 = spam_bow_transformer_6.transform(next_set['docs'])

    ### 청소년 유해 여부 분류
    harm_label = np.where(spam_train_set_6['label'] == '청소년유해', '청소년유해', '건전')
    harm_detect_model = MultinomialNB(alpha=0).fit(spam_docs_bow_6, harm_label)
    harm_predictions = harm_detect_model.predict(spam_docs_test_6)

    print(harm_predictions)
    i = 0
    j = 0
    while i < len(normal_predictions):
        if normal_predictions[i] == '비도박':
            normal_predictions[i] = harm_predictions[j]
            j += 1
        i += 1

    return normal_predictions


def binary_classifier_tfidf(df):
    global bow_transformer
    global docs_bow
    global train_set
    ### 정상 비정상 분류

    cv = CountVectorizer()
    bow_transformer = cv.fit(train_set['docs'])
    docs_bow = bow_transformer.transform(train_set['docs'])
    docs_test = bow_transformer.transform(df['docs'])
    normal_tfidf_transformer = TfidfTransformer().fit(docs_bow)
    normal_docs_tfidf = normal_tfidf_transformer.transform(docs_bow)
    normal_docs_test_tfidf = normal_tfidf_transformer.transform(docs_test)
    normal_label = np.where(train_set['label'] == '정상', '정상', '비정상')
    normal_detect_model = MultinomialNB(alpha=0.0001).fit(normal_docs_tfidf, normal_label)
    normal_predictions = normal_detect_model.predict(normal_docs_test_tfidf)
    normal_selector = []

    for i in range(len(normal_predictions)):
        if normal_predictions[i] == '비정상':
            normal_selector.append(i)

    next_set = df.iloc[normal_selector]
    # print(normal_predictions)
    ### 비정상데이터 Bag of words
    spam_train_set = df[df['label'] != '정상']
    spam_bow_transformer = cv.fit(spam_train_set['docs'])
    spam_docs_bow = spam_bow_transformer.transform(spam_train_set['docs'])
    spam_docs_test = spam_bow_transformer.transform(next_set['docs'])
    spam_tfidf_transformer = TfidfTransformer().fit(spam_docs_bow)
    spam_docs_tfidf = spam_tfidf_transformer.transform(spam_docs_bow)
    spam_docs_test_tfidf = spam_tfidf_transformer.transform(spam_docs_test)

    ### 대량 비대량 분류
    large_label = np.where(spam_train_set['label'] == '대량', '대량', '비대량')
    large_detect_model = MultinomialNB(alpha=0.0001).fit(spam_docs_tfidf, large_label)
    large_predictions = large_detect_model.predict(spam_docs_test_tfidf)
    # print(large_predictions)
    large_selector = []
    for i in range(len(large_predictions)):
        if large_predictions[i] == '비대량':
            large_selector.append(i)

    next_set = next_set.iloc[large_selector]

    i = 0
    j = 0
    while i < len(normal_predictions):
        if normal_predictions[i] == '비정상':
            normal_predictions[i] = large_predictions[j]
            j += 1
        i += 1

    ### 비대량데이터 Bag of words
    spam_train_set_2 = spam_train_set[spam_train_set['label'] != '대량']
    spam_bow_transformer_2 = cv.fit(spam_train_set_2['docs'])
    spam_docs_bow_2 = spam_bow_transformer_2.transform(spam_train_set_2['docs'])
    spam_docs_test_2 = spam_bow_transformer_2.transform(next_set['docs'])
    spam_tfidf_transformer_2 = TfidfTransformer().fit(spam_docs_bow_2)
    spam_docs_tfidf_2 = spam_tfidf_transformer_2.transform(spam_docs_bow_2)
    spam_docs_test_tfidf_2 = spam_tfidf_transformer_2.transform(spam_docs_test_2)

    ### 불법 합법 분류
    illegal_label = np.where(spam_train_set_2['label'] == '불법', '불법', '합법')
    illegal_detect_model = MultinomialNB(alpha=0.0001).fit(spam_docs_tfidf_2, illegal_label)
    illegal_predictions = illegal_detect_model.predict(spam_docs_test_tfidf_2)

    illegal_selector = []
    for i in range(len(illegal_predictions)):
        if illegal_predictions[i] == '합법':
            illegal_selector.append(i)

    next_set = next_set.iloc[illegal_selector]

    i = 0
    j = 0
    while i < len(normal_predictions):
        if normal_predictions[i] == '비대량':
            normal_predictions[i] = illegal_predictions[j]
            j += 1
        i += 1

    ### 비불법데이터 Bag of words
    spam_train_set_3 = spam_train_set_2[spam_train_set_2['label'] != '불법']
    spam_bow_transformer_3 = cv.fit(spam_train_set_3['docs'])
    spam_docs_bow_3 = spam_bow_transformer_3.transform(spam_train_set_3['docs'])
    spam_docs_test_3 = spam_bow_transformer_3.transform(next_set['docs'])
    spam_tfidf_transformer_3 = TfidfTransformer().fit(spam_docs_bow_3)
    spam_docs_tfidf_3 = spam_tfidf_transformer_3.transform(spam_docs_bow_3)
    spam_docs_test_tfidf_3 = spam_tfidf_transformer_3.transform(spam_docs_test_3)

    ### 도배 비도배 분류
    duplicate_label = np.where(spam_train_set_3['label'] == '도배', '도배', '비도배')
    duplicate_detect_model = MultinomialNB(alpha=0.0001).fit(spam_docs_tfidf_3, duplicate_label)
    duplicate_predictions = duplicate_detect_model.predict(spam_docs_test_tfidf_3)

    duplicate_selector = []
    for i in range(len(duplicate_predictions)):
        if duplicate_predictions[i] == '비도배':
            duplicate_selector.append(i)

    next_set = next_set.iloc[duplicate_selector]

    i = 0
    j = 0
    while i < len(normal_predictions):
        if normal_predictions[i] == '합법':
            normal_predictions[i] = duplicate_predictions[j]
            j += 1
        i += 1

    ### 비도배데이터 Bag of words
    spam_train_set_4 = spam_train_set_3[spam_train_set_3['label'] != '도배']
    spam_bow_transformer_4 = cv.fit(spam_train_set_4['docs'])
    spam_docs_bow_4 = spam_bow_transformer_4.transform(spam_train_set_4['docs'])
    spam_docs_test_4 = spam_bow_transformer_4.transform(next_set['docs'])
    spam_tfidf_transformer_4 = TfidfTransformer().fit(spam_docs_bow_4)
    spam_docs_tfidf_4 = spam_tfidf_transformer_4.transform(spam_docs_bow_4)
    spam_docs_test_tfidf_4 = spam_tfidf_transformer_4.transform(spam_docs_test_4)

    ### 홍보 여부 분류
    ad_label = np.where(spam_train_set_4['label'] == '홍보', '홍보', '비홍보')
    ad_detect_model = MultinomialNB().fit(spam_docs_tfidf_4, ad_label)
    ad_predictions = ad_detect_model.predict(spam_docs_test_tfidf_4)

    ad_selector = []
    for i in range(len(ad_predictions)):
        if ad_predictions[i] == '비홍보':
            ad_selector.append(i)

    next_set = next_set.iloc[ad_selector]

    i = 0
    j = 0
    while i < len(normal_predictions):
        if normal_predictions[i] == '비도배':
            normal_predictions[i] = ad_predictions[j]
            j += 1
        i += 1

    ### 비홍보데이터 Bag of words
    spam_train_set_5 = spam_train_set_4[spam_train_set_4['label'] != '홍보']
    spam_bow_transformer_5 = cv.fit(spam_train_set_5['docs'])
    spam_docs_bow_5 = spam_bow_transformer_5.transform(spam_train_set_5['docs'])
    spam_docs_test_5 = spam_bow_transformer_5.transform(next_set['docs'])
    spam_tfidf_transformer_5 = TfidfTransformer().fit(spam_docs_bow_5)
    spam_docs_tfidf_5 = spam_tfidf_transformer_5.transform(spam_docs_bow_5)
    spam_docs_test_tfidf_5 = spam_tfidf_transformer_5.transform(spam_docs_test_5)

    ### 도박 비도박 분류
    gamble_label = np.where(spam_train_set_5['label'] == '도박', '도박', '비도박')
    gamble_detect_model = MultinomialNB().fit(spam_docs_tfidf_5, gamble_label)
    gamble_predictions = gamble_detect_model.predict(spam_docs_test_tfidf_5)

    gamble_selector = []
    for i in range(len(gamble_predictions)):
        if gamble_predictions[i] == '비도박':
            gamble_selector.append(i)
            print(i)

    next_set = next_set.iloc[gamble_selector]

    i = 0
    j = 0
    while i < len(normal_predictions):
        if normal_predictions[i] == '비홍보':
            normal_predictions[i] = gamble_predictions[j]
            j += 1
        i += 1

    ### 비도박데이터 Bag of words
    print(spam_train_set_5)
    spam_train_set_6 = spam_train_set_5[spam_train_set_5['label'] != '도박']
    spam_bow_transformer_6 = cv.fit(spam_train_set_6['docs'])
    spam_docs_bow_6 = spam_bow_transformer_6.transform(spam_train_set_6['docs'])
    spam_docs_test_6 = spam_bow_transformer_6.transform(next_set['docs'])
    spam_tfidf_transformer_6 = TfidfTransformer().fit(spam_docs_bow_6)
    spam_docs_tfidf_6 = spam_tfidf_transformer_6.transform(spam_docs_bow_6)
    spam_docs_test_tfidf_6 = spam_tfidf_transformer_6.transform(spam_docs_test_6)

    ### 청소년 유해 여부 분류
    harm_label = np.where(spam_train_set_6['label'] == '청소년유해', '청소년유해', '건전')
    print(harm_label)
    harm_detect_model = MultinomialNB().fit(spam_docs_tfidf_6, harm_label)
    harm_predictions = harm_detect_model.predict(spam_docs_test_tfidf_6)

    print(harm_predictions)
    i = 0
    j = 0
    print(normal_predictions)
    while i < len(normal_predictions):
        if normal_predictions[i] == '비도박':
            normal_predictions[i] = harm_predictions[j]
            j += 1
        i += 1

    return normal_predictions


def binary_classifier_ovr(df):
    cv = CountVectorizer()
    bow_transformer = cv.fit(train_set['docs'])
    docs_bow = bow_transformer.transform(train_set['docs'])

    ### 정상 비비정상 분류
    normal_label = np.where(train_set['label'] == '정상', '정상', '비정상')
    normal_detect_model = MultinomialNB().fit(docs_bow, normal_label)

    ### 대량 비대량 분류
    large_label = np.where(train_set['label'] == '대량', '대량', '비대량')
    large_detect_model = MultinomialNB().fit(docs_bow, large_label)

    ### 불법 합법 분류
    illegal_label = np.where(train_set['label'] == '불법', '불법', '합법')
    illegal_detect_model = MultinomialNB().fit(docs_bow, illegal_label)

    ### 도배 비도배 분류
    duplicate_label = np.where(train_set['label'] == '도배', '도배', '비도배')
    duplicate_detect_model = MultinomialNB().fit(docs_bow, duplicate_label)

    ### 홍보 여부 분류
    ad_label = np.where(train_set['label'] == '홍보', '홍보', '비홍보')
    ad_detect_model = MultinomialNB().fit(docs_bow, ad_label)

    ### 도박 비도박 분류
    gamble_label = np.where(train_set['label'] == '도박', '도박', '비도박')
    gamble_detect_model = MultinomialNB().fit(docs_bow, gamble_label)

    ### 청소년 유해 여부 분류
    harm_label = np.where(train_set['label'] == '청소년유해', '청소년유해', '건전')
    harm_detect_model = MultinomialNB().fit(docs_bow, harm_label)

    normal_predictions = []
    for index, row in df.iterrows():
        normal_label_test = np.where(df['label'] == '정상', '정상', '비정상')

        normal_score = normal_detect_model.score(bow_transformer.transform(row['docs'], normal_label_test[index]))
        print(normal_score)

    return normal_predictions


print(classification_report(test_set['label'], binary_classifier(test_set)))
print(classification_report(test_set['label'], binary_classifier_tfidf(test_set)))
# print(classification_report(test_set['label'], binary_classifier_ovr(test_set)))
