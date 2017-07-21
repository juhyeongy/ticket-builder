import pandas as pd
import sqlite3
import json
import re
import gensim, logging
import numpy as np

from gensim.parsing.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Flatten, LSTM, Conv1D, MaxPooling1D, Embedding
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

###
# Hyperparams
#
W2V_MIN_COUNT = 2
W2V_SKIPGRAM_WINDOW = 4
W2V_DIM = 100
INPUT_MAX_LEN = 3000

TRAIN_EPOCH = 10
TRAIN_BATCH_SIZE = 128
TRAIN_SPLIT_RATIO = 0.20

p = re.compile("\{(.*)\}")
stopwords = stopwords.words('english')
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer('[a-zA-Z]{2,100}')

def extract_labels(raw):
    match = p.search(raw.replace('\n',''))
    if match:
        kvs = [tuple(line.split(':')) for line in match.group(1).split(',')]
        return {k.strip(): v.replace("'",'').strip() for (k, v) in kvs}
    return {}

def tokenize_raw_sentence(sentence):
    return [stemmer.stem(token) for token in tokenizer.tokenize(sentence) if token not in stopwords]

def build_padded_sequences(w2v_model, tokenized_sentences):
    index2word = w2v_model.wv.index2word
    word2index = {w:i for i,w in enumerate(index2word)}
    sequences = [[word2index[word] for word in words if word2index.has_key(word)] for words in tokenized_sentences]
    return pad_sequences(sequences, maxlen=INPUT_MAX_LEN, dtype='int32', padding='post', truncating='post', value=0.)

def build_embedding_layer(w2v_model):
    return Embedding(len(w2v_model.wv.index2word),
                     W2V_DIM,
                     weights=[w2v_model.wv.syn0],
                     input_length=INPUT_MAX_LEN,
                     trainable=False)

def build_conv_net(input_len, output_len, w2v_model):
    embedding_layer = build_embedding_layer(w2v_model)
    sequence_input = Input(shape=(input_len,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(output_len, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    return model

def load_model(name):
    model_config = pickle.load(open("./trained_models/{}-Config".format(name), "rb"))
    model_weights = pickle.load(open("./trained_models/{}-Weights".format(name), "rb"))
    model = Model.from_config(model_config)
    model.set_weights(model_weights)
    return model

def persist_models(clf_y_pairs):
    for name, clf, y in clf_y_pairs:
        pickle.dump(clf.get_config(), open("./trained_models/{}-Config".format(name), "wb"))
        pickle.dump(clf.get_weights(), open("./trained_models/{}-Weights".format(name), "wb"))
        pickle.dump(y.columns, open("./trained_models/{}-Columns".format(name), "wb"))

def persist_word2vec(w2v):
    w2v.save("./trained_models/Word2Vector")

def load_word2vec():
    return Word2Vec.load("./trained_models/Word2Vector")


###
# Read in crawled bugzilla data.
#
conn = sqlite3.connect('../crawler/data/result.db')
df = pd.read_sql('select result from resultdb_bugzilla_with_labels_with_priority', con=conn)
df['row'] = df['result'].map(lambda row: json.loads(row))
label_sets = df['row'].map(lambda row: extract_labels(row['labels']))


###
# Prepare raw bugzilla dataset for predictors and response
#
df['raw_sentence'] = df['row'].map(lambda row: row['body'])
df['y_category'] = label_sets.map(lambda label_set: label_set['category'])
df['y_component'] = label_sets.map(lambda label_set: label_set['component'])
df['y_severity'] = df['row'].map(lambda row: row['bug_severity'])
df['y_priority'] = df['row'].map(lambda row: row['priority'])
df['y_cftype'] = df['row'].map(lambda row: row['cf_type'])
df = df.drop(['result','row'], axis=1)

###
# Clean, stemminize corpus and build word2vec model to vectorize token against to
#
tokenized_sentences = df['raw_sentence'].map(lambda raw_sentence: tokenize_raw_sentence(raw_sentence))
w2v_model = Word2Vec(tokenized_sentences, min_count=W2V_MIN_COUNT, size=W2V_DIM, window=W2V_SKIPGRAM_WINDOW)

###
# Prepare training dataset by converting tokenized senteces into padded sequence of word2vec indexes
#
X = build_padded_sequences(w2v_model, tokenized_sentences)
y_category = pd.get_dummies(df['y_category'])
y_component = pd.get_dummies(df['y_component'])
y_severity = pd.get_dummies(df['y_severity'])
y_priority = pd.get_dummies(df['y_priority'])
y_cftype = pd.get_dummies(df['y_cftype'])

###
# Prepare training dataset by converting tokenized senteces into padded sequence of word2vec indexes
#
category_clf = build_conv_net(INPUT_MAX_LEN, y_category.shape[1], w2v_model)
component_clf = build_conv_net(INPUT_MAX_LEN, y_component.shape[1], w2v_model)
severity_clf = build_conv_net(INPUT_MAX_LEN, y_severity.shape[1], w2v_model)
priority_clf = build_conv_net(INPUT_MAX_LEN, y_priority.shape[1], w2v_model)
cftype_clf = build_conv_net(INPUT_MAX_LEN, y_cftype.shape[1], w2v_model)

clf_y_pairs = [
    ('Category-Classifier', category_clf, y_category),
    ('Component-Classifier', component_clf, y_component),
    ('Severity-Classifier', severity_clf, y_severity),
    ('Priority-Classifier', priority_clf, y_priority),
    ('Cftype-Classifier', cftype_clf, y_cftype)
]

for name, clf, y in clf_y_pairs:
    x_train, x_test, y_train, y_test = train_test_split(X, y.as_matrix(), test_size=TRAIN_SPLIT_RATIO, random_state=42)
    print 'Training {} Now...'.format(name)
    clf.fit(x_train, y_train, batch_size=TRAIN_BATCH_SIZE, epochs=TRAIN_EPOCH, validation_data=(x_test, y_test))

###
# Persisting models and word_embedding
# persist_models(clf_y_pairs)
# persist_word2vec(w2v_model)
# load_word2vec()


inquery = """
    Bhavesh reported Astro becoming VERY slow for a system that had been up for 7+days
    Investigation shows
    Current state:
    ===========
    top - 18:43:54 up 5 days, 19:25,  3 users,  load average: 3.99, 2.94, 2.21
    Tasks: 404 total,   3 running, 400 sleeping,   0 stopped,   1 zombie
    %Cpu(s): 67.9 us,  5.1 sy,  0.0 ni, 25.1 id,  0.7 wa,  0.0 hi,  1.3 si,  0.0 st
    KiB Mem : 97.4/12303828 [|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||   ]
    KiB Swap: 40.4/3903484  [||||||||||||||||||||||||||||||||||||||||                                                            ]
    which is pretty high. closer digging shows avmanager processes (4x) consuming close to 5.3 GB...which is somewhat concerning
    the four AV processes are all consuming over 1 GB each, on average, which seems like a lot)
    when we restarted avmanager and the connector things seemed to settle down for a while, but again got sluggish some time later.
    the logs are big but i do see frequent periods where the avamnager logs are full of things like this:
    [2017-02-17 23:37:57 UTC #19794]  INFO -- P19794R8278: 10.139.88.1:3443 <- /xms/identity_identities #<Thread:0x0000000330ea28>
    [2017-02-17 23:37:57 UTC #19794]  INFO -- P19794R8286: 10.139.88.1:3443 <- /xms/identity_identities #<Thread:0x007ff9746a4a80>
    ------------+--------------------------------------+--------------+----------+-----------+--------
    management | 5a6818df-a009-4192-bf9f-c6a30d8b64d6 | 10.139.84.21 |          | 127.0.0.2 |   1000
    tenant     | 3e28951b-8082-4925-b053-54d4274ec744 | 10.139.88.1  |          | 127.0.0.3 |   1001
    Hydra Tenant Administration = https://hydra-enzo-erqe.cp.horizon.vmware.com/horizonadmin
    Tenant/Customer = customer@vxrail.lab/VMW@re123
    Tenant Admin = tenantadmin@pod22.net/VMW@re123
    vSphere username is :  enzoadmin@vsphere.local
    vSphere password is : VMW@re123
    vCenter IP: 10.139.84.12 (In order for webclient to work add 10.157.1.1 as DNS server)
    im going to attach logs to this bug as a convenient location to put them. slowness seem during the day of 17th Feb 2017.... primarily around 10am-->4pm or so but the actual times i dont think were restricted to that...
    as i keep digging i will update this bug further. sorry for the vagueness...
"""
converted_inquery = build_padded_sequences(w2v_model, [tokenize_raw_sentence(inquery)])

for name, clf, y in clf_y_pairs:
    label_index = np.argmax(clf.predict(converted_inquery), axis=1)
    label = y.columns.tolist()[label_index]
    print '"{}" labled the input as "{}"'.format(name, label)
