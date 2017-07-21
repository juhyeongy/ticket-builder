import pandas as pd
import json
import nltk
import statsmodels.api as sm
import re
import numpy as np
import pickle

from keras.models import Sequential, Model
from sklearn.externals import joblib
from flask import Flask, request
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.parsing.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

INPUT_MAX_LEN = 3000

p = re.compile("\{(.*)\}")
stopwords = stopwords.words('english')
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer('[a-zA-Z]{2,100}')

def load_word2vec():
    return Word2Vec.load("./trained_models/Word2Vector")

def load_model_and_columns(name):
    model_config = pickle.load(open("./trained_models/{}-Config".format(name), "rb"))
    model_weights = pickle.load(open("./trained_models/{}-Weights".format(name), "rb"))
    columns = pickle.load(open("./trained_models/{}-Columns".format(name), "rb"))
    model = Model.from_config(model_config)
    model.set_weights(model_weights)
    return (model, columns)

def tokenize_raw_sentence(sentence):
    return [stemmer.stem(token) for token in tokenizer.tokenize(sentence) if token not in stopwords]

def build_padded_sequences(w2v_model, tokenized_sentences):
    index2word = w2v_model.wv.index2word
    word2index = {w:i for i,w in enumerate(index2word)}
    sequences = [[word2index[word] for word in words if word2index.has_key(word)] for words in tokenized_sentences]
    return pad_sequences(sequences, maxlen=INPUT_MAX_LEN, dtype='int32', padding='post', truncating='post', value=0.)

model_names = [
  'Category-Classifier',
  'Component-Classifier',
  'Severity-Classifier',
  'Priority-Classifier',
  'Cftype-Classifier'
]

models = [(name, load_model_and_columns(name)) for name in model_names]
w2v_model = load_word2vec()

app = Flask(__name__)

@app.route("/api/preds/bugzilla-tags",  methods=['POST'])
def predict():
  content = request.json
  text = content['text'].strip()

  if len(text) < 30:
    return json.dumps({})

  converted_text = build_padded_sequences(w2v_model, [tokenize_raw_sentence(text)])
  tags = {}

  for (name, (model, columns)) in models:
      name = name.lower().split('-')[0]
      label_index = np.argmax(model.predict(converted_text), axis=1)
      label = columns.tolist()[label_index]
      tags[name] = label

  return json.dumps(tags)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
