import numpy as np
np.random.seed(42)
import pandas as pd
import os,sys,time,datetime,gc

from sklearn.metrics import *

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Dropout
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from sklearn.model_selection import StratifiedKFold

import tensorlayer as tl

import warnings
import config
import utils

os.environ['OMP_NUM_THREADS'] = '4'

strategy = 'bi-gru-attention'

cs_delete_file = '%s/raw/内容联系方式样本_0716.xlsx' % config.DataBaseDir
pos_58_file = '%s/raw/58_2d_55-85_positive_labeled.csv' % config.DataBaseDir
neg_58_file = '%s/raw/58_2d_25-45_negative_labeled.csv' % config.DataBaseDir

max_features = 10000
maxlen = 150
batch_size = 32
epochs = 2

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim

def bi_gru_attention(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, config.embedding_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.25)(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.25)(x)
    outp = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

## load data
with utils.timer('Load data'):
    data_1 = utils.load_cs_deleted_data(cs_delete_file)
    print('target ratio: ')
    print(data_1['label'].value_counts())
    data_2 = utils.load_58_data(pos_58_file)
    print(data_2['label'].value_counts())
    data_3 = utils.load_58_data(neg_58_file)
    print(data_3['label'].value_counts())
    data = pd.concat([data_1, data_2, data_3], axis=0, ignore_index=True)
    DebugDir = '%s/debug' % config.DataBaseDir
    if (os.path.exists(DebugDir) == False):
        os.makedirs(DebugDir)
    del data_3, data_2, data_1
    gc.collect()

## load word2vec lookup table
with utils.timer('Load word vector'):
    word2vec = tl.files.load_npy_to_any(name='%s/model/word2vec_post_text_3d.npy' % config.DataBaseDir)
    print('embedding word size: %s' % len(word2vec))

## representation
with utils.timer('representation'):
    X_words = np.array(data['text'].apply(utils.cut))
    y = np.array(data['label'])

    tokenizer = text.Tokenizer(num_words= max_features)
    tokenizer.fit_on_texts(X_words)
    X = tokenizer.texts_to_sequences(X_words)
    del X_words
    gc.collect()
    X = sequence.pad_sequences(X, maxlen= maxlen)
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, config.embedding_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

with utils.timer('Train'):
    for s in range(config.train_times):
        s_start = time.time()
        train_pred = np.zeros(len(X))

        skf = StratifiedKFold(config.kfold, random_state= 2018 * s, shuffle=False)

        for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
            f_start = time.time()

            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            print('shape of train data:')
            print(X_train.shape)
            print('shape of test data:')
            print(y_train.shape)

            model = bi_gru_attention(embedding_matrix)
            RocAuc = RocAucEvaluation(validation_data=(X_valid, y_valid), interval=1)
            hist = model.fit(X_train, y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             validation_data=(X_valid, y_valid),
                             callbacks=[RocAuc], verbose=2)
            valid_pred_proba = model.predict(X_valid, batch_size= batch_size)
            valid_pred_label = utils.proba2label(valid_pred_proba)
            valid_auc = roc_auc_score(y_valid, valid_pred_proba)
            valid_precision = precision_score(y_valid, valid_pred_label)
            valid_recall = recall_score(y_valid, valid_pred_label)

            train_pred[valid_index] = valid_pred_proba

            f_end = time.time()
            print('#%s[fold %s]: auc %.6f, precision %.6f, recall %.6f, took %s[s]' % (s, fold, valid_auc, valid_precision, valid_recall, int(f_end - f_start)))

        auc = roc_auc_score(y, train_pred)
        precision = precision_score(y, utils.proba2label(train_pred))
        recall = recall_score(y, utils.proba2label(train_pred))

        s_end = time.time()
        print('\n===================================================')
        print('#%s: auc %.6f, precision %.6f, recall %.6f, took %s[s]' % (s, auc, precision, recall, int(s_end - s_start)))
        print('===================================================\n')
