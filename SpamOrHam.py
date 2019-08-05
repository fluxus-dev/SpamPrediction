import pandas as pd
import os.path


if not os.path.isfile('new_blog_spam.txt'):
    f = open("new_blog_spam.txt", 'w', encoding='UTF8')
    for line in open('blog_spam.txt', encoding='UTF8'):
        temp = line.split('\t')
        new_line = '\t'.join(temp[:4])+' '.join(temp[4:])
        f.write(new_line)
    f.close()

df = pd.read_csv("new_blog_spam.txt", sep='\t', names=["type", "label", "docid", "docs"])
print(df.head())

print(df.groupby("label").describe())

df['length'] = df['docs'].apply(len)

import matplotlib.pyplot as plt
import seaborn as sns
#
# plt.hist(df['length'])
# plt.show()

print(df['length'].describe())

print(df[df['length']==1484]['docs'].iloc[0])