import random
import happybase
import nltk
import numpy as np
from keras import layers
from keras import losses
from keras import metrics
from keras import models
from keras import optimizers
from konlpy.tag import Okt

keyword = ""
keyword_utf8 = keyword.encode("utf-8")

model = models.Sequential()
selected_words = []

okt = Okt()


def tokenize(doc):
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]


def get_data():
    global selected_words
    global keyword_utf8
    raw_data = []
    pos_data = []
    neg_data = []
    for data in table.scan(row_prefix=keyword_utf8):

        if int(data[1][b'data:score'].decode('UTF-8')) > 4:

            temp_data = [data[1][b'data:title'].decode("utf-8") + data[1][b'data:fulltext'].decode("utf-8"), 1]
            pos_data.append(temp_data)
            # raw_data.append(temp_data)
        elif int(data[1][b'data:score'].decode('UTF-8')) < 3:
            temp_data = [data[1][b'data:title'].decode("utf-8") + data[1][b'data:fulltext'].decode("utf-8"), 0]
            neg_data.append(temp_data)
            # raw_data.append(temp_data)
    # random.shuffle(raw_data)

    random.shuffle(pos_data)
    print(len(neg_data))
    raw_data += pos_data[:len(neg_data)]
    raw_data += neg_data
    print(len(raw_data))
    random.shuffle(raw_data)
    base = int(len(raw_data) * 0.95)
    train_data = raw_data[:base]
    test_data = raw_data[base:]
    train_docs = [(tokenize(row[0]), row[1]) for row in train_data]
    test_docs = [(tokenize(row[0]), row[1]) for row in test_data]
    tokens = [t for d in train_docs for t in d[0]]
    text = nltk.Text(tokens, name='NMSC')
    selected_words = [f[0] for f in text.vocab().most_common(5000)]
    train_x = [term_frequency(d, selected_words) for d, _ in train_docs]
    test_x = [term_frequency(d, selected_words) for d, _ in test_docs]
    train_y = [c for _, c in train_docs]
    test_y = [c for _, c in test_docs]
    x_train = np.asarray(train_x).astype('float32')
    x_test = np.asarray(test_x).astype('float32')
    y_train = np.asarray(train_y).astype('float32')
    y_test = np.asarray(test_y).astype('float32')
    print(training_models(x_train, y_train, x_test, y_test))


def term_frequency(doc, selected_words):

    return [doc.count(word) for word in selected_words]


def training_models(x_train, y_train, x_test, y_test):
    global model
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=0.005),
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    model.fit(x_train, y_train, epochs=3, batch_size=64)
    results = model.evaluate(x_test, y_test)
    model.save('tour_nlp_model.h5')
    return results


def predict_pos_neg(review):
    global selected_words
    token = tokenize(review)
    tf = term_frequency(token, selected_words)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    # if score < 0.05:
    #     print("[{}...]는 {:.2f}% 확률로 부정 리뷰\n".format(review[:120], (1 - score) * 100))
    if score > 0.95 :
        print("[{}...]는 {:.2f}% 확률로 긍정 리뷰\n".format(review[:80], score * 100))
    elif score < 0.05:
        print("[{}...]는 {:.2f}% 확률로 부정 리뷰\n".format(review[:100], (1 - score) * 100))
    else:
        print("[{}...]는 중립 리뷰\n".format(review[:80]))



if __name__ == '__main__':
    tableName = 'tripadvisor'
    connection = happybase.Connection('10.80.18.23', port=9090)
    table = connection.table(tableName)
    import os.path

    if os.path.isfile('tour_nlp_model.h5'):
        get_data()
    else:
        model = models.load_model('tour_nlp_model_temp.h5')

    predict_pos_neg("별로네요. 사람만 많고")
    predict_pos_neg("팔만대장경이 있는 곳이죠~~ 절 가는 길이 아름다운 곳입니다!! 아름다운 절 꼭 방문해보세요~~")
    predict_pos_neg("추천하지 않음, 상업화되어 있음")
    predict_pos_neg("개발이 망쳐놓은 장소 과거 선샤인 호텔만 있을 땐 정말 좋았다. 바다도 아름답고 여유가 느껴지던 곳. 그러나 다시 찾은 함덕은 모텔급 숙박시설의 난립으로 최악의 환경이 되었다. 아쉽다.")
    predict_pos_neg("세계적인 습지 세계적인 습지이다 순천만 정원과 스카이 큐브로 연결되어 있어 쉽게 갈 수 있다 볼거리도 많아 순천 오시는 분들은 방문하길 권한다")
    predict_pos_neg("도심에 지친 사람들이 가면 너무나 좋을 곳. 저녁에는 노을이 매일매일 다르게 예쁘다.")
    predict_pos_neg("비싸고 볼게 없어요.")
    predict_pos_neg("덥긴 했지만 볼만했습니다.")
    predict_pos_neg("볼만하긴 했지만 더웠습니다.")



