import numpy as np
np.random.seed(2018)
import pandas as pd
import os,sys,time,datetime,gc
import shutil

from sklearn.metrics import *

import tensorflow as tf
import tensorlayer as tl

from keras.models import Model
from keras.layers import Input, Dense, SpatialDropout1D, concatenate
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, LSTM
from keras.callbacks import Callback
import keras.backend as K
from keras.models import model_from_json

from tensorflow.python.util import compat
from sklearn.model_selection import StratifiedKFold

import config
import utils

os.environ['OMP_NUM_THREADS'] = '4'

strategy = 'bi-lstm'

cs_delete_file_1 = '%s/raw/内容联系方式样本_0716.xlsx' % config.DataBaseDir
cs_delete_file_2 = '%s/raw/内容联系方式样本_0725.xlsx' % config.DataBaseDir
pos_58_file = '%s/raw/58_2d_55-85_positive_labeled.csv' % config.DataBaseDir
neg_58_file = '%s/raw/58_2d_25-45_negative_labeled.csv' % config.DataBaseDir
test_file = '%s/raw/test.txt' % config.DataBaseDir

model_version = 1
max_features = 20000
maxlen = 150
batch_size = 32
epochs = 12
#datestr = datetime.datetime.now().strftime("%Y%m%d")
datestr = '20180803'

## hold the resources in the first place
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=False)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)

def export_model(model_dir, model_version, model):
    ''''''
    path = os.path.dirname(os.path.abspath(model_dir))
    if os.path.isdir(path) == False:
        os.makedirs(path)

    export_path = os.path.join(
        compat.as_bytes(model_dir),
        compat.as_bytes(str(model_version)))

    if os.path.isdir(export_path) == True:
        shutil.rmtree(export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    model_input = tf.saved_model.utils.build_tensor_info(model.input)
    model_output = tf.saved_model.utils.build_tensor_info(model.output)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'inputs': model_input},
            outputs={'output': model_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    )

    legacy_init = tf.group(tf.tables_initializer(), name='legacy_init_op')

    # Initialize global variables and the model
    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with K.get_session() as sess:
        # sess.run(init_op)
        builder.add_meta_graph_and_variables(
            sess= sess,
            tags= [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict':
                    prediction_signature,
                # tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
            }, legacy_init_op= legacy_init)

        builder.save()

def bi_lstm_model(spatial_dropout = 0.5, dropout= 0.25, recurrent_dropout= 0.25):
    inp = Input(shape=(maxlen, config.embedding_size))
    x = SpatialDropout1D(spatial_dropout)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout= dropout, recurrent_dropout= recurrent_dropout))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs= inp, outputs=outp)

    return model

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            print('predict positive %s' % np.sum(utils.proba2label(y_pred)))
            auc = roc_auc_score(self.y_val, y_pred)
            precision = precision_score(self.y_val, utils.proba2label(y_pred))
            recall = recall_score(self.y_val, utils.proba2label(y_pred))
            print("\n ROC-AUC - epoch: %d - auc: %.6f - precision %.6f - recall %.6f\n" % (epoch+1, auc, precision, recall))

def LoadTrainData():
    ''''''
    with utils.timer('Load train data'):
        data_1 = utils.load_cs_deleted_data(cs_delete_file_1)
        print('target ratio: ')
        print(data_1['label'].value_counts())
        data_2 = utils.load_58_data(pos_58_file)
        print(data_2['label'].value_counts())
        data_3 = utils.load_58_data(neg_58_file)
        print(data_3['label'].value_counts())
        data_4 = utils.load_cs_deleted_data(cs_delete_file_2)
        print(data_4['label'].value_counts())
        data = pd.concat([data_1, data_2, data_3, data_4], axis=0,ignore_index=True)
        del data_4, data_3, data_2, data_1
        #data = data[:int(0.6 * len(data))]
        gc.collect()
    return data

def LoadTestData():
    ''''''
    with utils.timer('Load test data'):
        data, uid_list, info_id_list = utils.load_test_data(test_file)
    return data, uid_list, info_id_list

def LoadWord2Vec():
    with utils.timer('Load word vector'):
        word2vec = tl.files.load_npy_to_any(name='%s/model/word2vec/w2v_sgns_500_post_text_7d_%s.npy' % (config.DataBaseDir, datestr))
    return word2vec

def WordRepresentation(texts, word2vec):
    ''''''
    with utils.timer('representation'):
        ## padding
        X = []
        indexes = []
        for i in range(len(texts)):
            text = texts[i]
            words = utils.cut(text)
            if (len(words) == 0):
                continue
            if (len(words) < maxlen):
                X.append(['_UNK'] * (maxlen - len(words)) + words) # ahead padding in default mode
            else:
                X.append(words[:maxlen])
            indexes.append(i)
        ## word2vec
        X = np.array([[word2vec.get(w, word2vec['_UNK']) for w in wv] for wv in X])
    return X,  indexes

def SaveCheckpoint(model, outputdir):
    ''''''
    with utils.timer('save ckpt model'):
        # save model schema
        model_json = model.to_json()
        with open("%s/%s.json" % (outputdir, strategy), "w") as o_file:
            o_file.write(model_json)
        o_file.close()
        # save model weights
        model.save_weights("%s/%s.h5" % (outputdir, strategy))

def LoadCheckpoint(inputdir):
    ''''''
    with utils.timer('load ckpt model'):
        # load model schema
        with open("%s/%s.json" % (inputdir, strategy), "r") as i_file:
            loaded_model_json = i_file.read()
        i_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load model weights
        loaded_model.load_weights("%s/%s.h5" % (inputdir, strategy))

    return loaded_model

def train(X, y, model, outputdir):
    ''''''
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    with utils.timer('Train'):
        for s in range(config.train_times):
            s_start = time.time()
            train_pred = np.zeros((len(X), 1))

            skf = StratifiedKFold(config.kfold, random_state=2018 * s, shuffle=False)

            for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
                f_start = time.time()

                X_train, X_valid = X[train_index], X[valid_index]
                y_train, y_valid = y[train_index], y[valid_index]

                print('shape of train data:')
                print(X_train.shape)
                print('shape of valid data:')
                print(X_valid.shape)

                RocAuc = RocAucEvaluation(validation_data=(X_valid, y_valid), interval=1)
                model.fit(X_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(X_valid, y_valid),
                                 callbacks=[RocAuc], verbose=2)
                valid_pred_proba = model.predict(X_valid, batch_size=batch_size)
                train_pred[valid_index] = valid_pred_proba

                f_end = time.time()
                print('#%s[fold %s]: took %s[s]' % (s, fold, int(f_end - f_start)))

                del X_train, X_valid, y_train, y_valid
                gc.collect()

                break
    ## save checkpoint
    SaveCheckpoint(model, outputdir)

def test(X, model, outputdir, auxinfo):
    ''''''
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    pred_test = model.predict(X, batch_size= batch_size).flatten()
    print(pred_test.shape)
    auxinfo['text'] = np.array([text.replace(',', ' ') for text in auxinfo['text']])
    outdf = pd.DataFrame({'uid': auxinfo['uid'], 'info_id': auxinfo['info_id'], 'text': auxinfo['text'], 'predict_proba': pred_test})
    outdf[['uid', 'info_id', 'text', 'predict_proba']].to_csv('%s/%s_%s.csv' % (outputdir, strategy, datestr),header=True, index=False, encoding='utf-8')

if __name__ == '__main__':
    ''''''
    if(len(sys.argv) != 2):
        print('%s phase[train|export|test]' % sys.argv[0])
        sys.exit(1)
    phase = sys.argv[1]

    # checkpoint dir
    ckptdir = '%s/model/%s/checkpoint' % (config.DataBaseDir, strategy)
    if(os.path.exists(ckptdir) == False):
        os.makedirs(ckptdir)
    # infer dir
    inferdir = '%s/model/%s/infer' % (config.DataBaseDir, strategy)
    if(os.path.exists(inferdir) == False):
        os.makedirs(inferdir)
    # test dir
    testdir = '%s/model/%s/test' % (config.DataBaseDir, strategy)
    if(os.path.exists(testdir) == False):
        os.mkdir(testdir)

    if(phase == 'train'):
        data = LoadTrainData()
        word2vec = LoadWord2Vec()
        X, indexes = WordRepresentation(data['text'].values, word2vec)
        y = data['label'].values[indexes]
        del data, word2vec
        gc.collect()
        model = bi_lstm_model()
        model.summary()
        train(X, y, model, ckptdir)
    elif(phase == 'export'):
        K.set_learning_phase(0) ##!!! need to be set before loading model
        model = LoadCheckpoint(ckptdir)
        model.summary()
        export_model(inferdir, model_version, model)
    elif(phase == 'test'):
        data, uid_list, info_id_list = LoadTestData()
        word2vec = LoadWord2Vec()
        X, indexes = WordRepresentation(data, word2vec)
        del word2vec
        gc.collect()
        model = LoadCheckpoint(ckptdir)
        model.summary()
        test(X, model, testdir, {'uid': uid_list[indexes], 'info_id': info_id_list[indexes], 'text': data[indexes]})
