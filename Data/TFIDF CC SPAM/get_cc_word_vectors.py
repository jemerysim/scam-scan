import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_pickle(r'C:\Users\Wong Wei Kang\Github Projects\scam-scan\src\data\tokenized_text_SPAM.pickle')
df.head(3)

from sklearn.model_selection import train_test_split

variables = df.columns[3:]
target_col = df.columns[1]
X = df[variables]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, shuffle=True, stratify = y)

X_train.head()

print(X.columns)
X_train.index
X_test.index

from gensim.models.keyedvectors import KeyedVectors
import numpy as np
cc = KeyedVectors.load_word2vec_format(r'C:\Users\Wong Wei Kang\Github Projects\scam-scan\crawl-300d-2M.vec', limit=400000)
vocab = cc.vocab

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

docs = X_train["cleaned_text"].values
# labels = df.spam.values
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)

vocab_size = len(t.word_index) + 1 
vocab_map = t.word_index
X_train_seq = t.texts_to_sequences(X_train['cleaned_text'])
X_test_seq = t.texts_to_sequences(X_test['cleaned_text'])

max_length =max( max(map(len, X_test_seq)), max(map(len, X_train_seq)))


X_train_pad = pad_sequences(X_train_seq, padding='post', maxlen = max_length)
X_test_pad = pad_sequences(X_test_seq, padding='post', maxlen = max_length)
X_train_pad.shape
X_test_pad.shape

cc_word_vector_matrix = np.zeros((len(vocab_map) , 300))

print(cc_word_vector_matrix.shape)

for word, index in vocab_map.items():
    try:
        vector = cc.word_vec(word)
        cc_word_vector_matrix[index] = vector
    except:
        print(word)

print(cc_word_vector_matrix.shape)