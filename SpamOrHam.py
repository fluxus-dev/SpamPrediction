import os.path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

if not os.path.isfile('new_blog_spam.txt'):
    f = open("new_blog_spam.txt", 'w', encoding='UTF8')
    for line in open('blog_spam.txt', encoding='UTF8'):
        temp = line.split('\t')
        new_line = '\t'.join(temp[:4]) + ' ' + ' '.join(temp[4:])
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

test_set = df[df['type'] == 'ts']

print(train_set.head())

cv = CountVectorizer()
bow_transformer = cv.fit(train_set['docs'])
docs_bow = bow_transformer.transform(train_set['docs'])
print(docs_bow.shape)
# x_train


spam_detect_model = MultinomialNB().fit(docs_bow, train_set['label'])

all_predictions = spam_detect_model.predict(docs_bow)
print(classification_report(train_set['label'], all_predictions))
print('\n')

m_confusion_test = confusion_matrix(train_set['label'], all_predictions)
print(pd.DataFrame(data=m_confusion_test))
# binary_train_set.loc[binary_train_set.label != '정상', 'label'] = '비정상'

bow_transformer = cv.fit(train_set['docs'])
docs_bow = bow_transformer.transform(train_set['docs'])

###tf-idf 활용
tfidf_transformer = TfidfTransformer().fit(docs_bow)
docs_tfidf = tfidf_transformer.transform(docs_bow)

tfidf_detect_model = MultinomialNB().fit(docs_tfidf, train_set['label'])
tfidf_predictions = tfidf_detect_model.predict(docs_tfidf)

print(classification_report(train_set['label'], tfidf_predictions))
print('\n')

### 정상 비정상 분류
normal_label = np.where(train_set['label'] == '정상', '정상', '비정상')
normal_detect_model = MultinomialNB().fit(docs_bow, normal_label)
normal_predictions = normal_detect_model.predict(docs_bow)
print(normal_predictions)
print(classification_report(normal_label, normal_predictions))
print('\n')
normal_confusion_test = confusion_matrix(normal_label, normal_predictions)
print(pd.DataFrame(data=normal_confusion_test, columns = ['예측 정상', '예측 비정상'],
            index = ['실제 정상', '실제 비정상']))

### 대량 비대량 분류
large_label = np.where(train_set['label'] == '대량', '대량', '비대량')
large_detect_model = MultinomialNB().fit(docs_bow, large_label)
large_predictions = large_detect_model.predict(docs_bow)
print(large_predictions)
print(classification_report(large_label, large_predictions))
print('\n')

### 도박 비도박 분류
gamble_label = np.where(train_set['label'] == '도박', '도박', '비도박')
gamble_detect_model = MultinomialNB().fit(docs_bow, gamble_label)
gamble_predictions = gamble_detect_model.predict(docs_bow)
print(classification_report(gamble_label, gamble_predictions))
print('\n')

### 도배 비도배 분류
duplicate_label = np.where(train_set['label'] == '도배', '도배', '비도배')
duplicate_detect_model = MultinomialNB().fit(docs_bow, duplicate_label)
duplicate_predictions = duplicate_detect_model.predict(docs_bow)
print(classification_report(duplicate_label, duplicate_predictions))
print('\n')

### 불법 합법 분류
illegal_label = np.where(train_set['label'] == '불법', '불법', '합법')
illegal_detect_model = MultinomialNB().fit(docs_bow, illegal_label)
illegal_predictions = illegal_detect_model.predict(docs_bow)
print(classification_report(illegal_label, illegal_predictions))
print('\n')

### 청소년 유해 여부 분류
harm_label = np.where(train_set['label'] == '청소년유해', '청소년유해', '건전')
harm_detect_model = MultinomialNB().fit(docs_bow, harm_label)
harm_predictions = harm_detect_model.predict(docs_bow)
print(classification_report(harm_label, harm_predictions))
print('\n')

### 홍보 여부 분류
ad_label = np.where(train_set['label'] == '홍보', '홍보', '비홍보')
ad_detect_model = MultinomialNB().fit(docs_bow, ad_label)
ad_predictions = ad_detect_model.predict(docs_bow)
print(classification_report(ad_label, ad_predictions))
print('\n')

test_data = test_set.loc[0, 'docs']
print(test_data)
print(ad_detect_model.predict(bow_transformer.transform([test_data])))
