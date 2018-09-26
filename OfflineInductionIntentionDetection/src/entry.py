# entry of this project
#
# This is a simplified implementation version of Muti-turn Recurrent Neural Network oriented on intention classification with contextual infomation,
# which is referenced from http://giusepperizzo.github.io/publications/Mensio_Rizzo-HQA2018.pdf, while the dataset used here is come
# from https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/.
#
# Intention of offline induction detection within inter-message scenario is our final goal of this project, which is to be developed within the near future.
#
# Created by yuanpingzhou at 9/25/18

# common imports
import argparse
import data_utils
import psutil, os
import numpy as np
import sys, gc
from nltk.tokenize import word_tokenize
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import nltk
nltk.download('punkt')

# tf imports
import tensorflow as tf

# keras imports
from keras.models import Model
from keras.layers import Input, Dense, SpatialDropout2D, TimeDistributed, Bidirectional, LSTM
from keras.layers import concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.callbacks import Callback, EarlyStopping
import keras.backend as K

# custom imports
import utils
import config

# mem usage for debugging
process = psutil.Process(os.getpid())
def _print_memory_usage():
    ''''''
    print('\n---- Current memory usage %sM ----\n' % int(process.memory_info().rss/(1024*1024)))

# configuration for GPU resources
with K.tf.device('/device:GPU:0'):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.8, allow_growth=False)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)

##
def get_coefs(word, *arr):
    return (word, np.asarray(arr, dtype='float32'))

## load word embedding
def load_word_embedding_vectors(f, corpus):
    k = 0
    EmbeddingDict = {}
    with open(f, 'r', encoding= 'utf-8') as i_file:
        for line in i_file:
            #if(k == 10000):
            #   break
            words = line.rstrip().rsplit(' ')
            if(words[0] in corpus):
                w, coe_vec= get_coefs(*words)
                EmbeddingDict[w] = coe_vec
            if(k % 10000 == 0):
                print('%s done.' % k)
            k += 1
    i_file.close()
    return EmbeddingDict

## a faster version of loading word embedding
def load_word_embedding_vectors_1(f):
    with open(f, 'r', encoding= 'utf-8') as i_file:
        embedding_dict = dict([get_coefs(*line.rstrip().rsplit(' ')) for line in i_file])
    i_file.close()
    return embedding_dict

## Muti-turn QA recurrent neural network framwork, referenced by http://giusepperizzo.github.io/publications/Mensio_Rizzo-HQA2018.pdf
def get_network(sentence_dim= 128,
                dialogue_dim= 64,
                input_spatial_dropout= 0.5,
                rnn_vertical_dropout= 0.25,
                rnn_horizontal_dropout= 0.25,
                print_network= True):

    # input
    input_layer = Input(shape=(config.max_sentence, config.max_word, config.word_embedding_dim))

    # sentence level
    x = SpatialDropout2D(input_spatial_dropout)(input_layer)
    encoded_sentence = TimeDistributed(Bidirectional(LSTM(sentence_dim, dropout= rnn_vertical_dropout, recurrent_dropout= rnn_horizontal_dropout)))(x)

    # dialogue level
    encoded_dialogue = LSTM(dialogue_dim)(encoded_sentence)

    ## TODO
    ## add max/avg pooling layers or dropout layers here for better performance

    # output
    output_layer = Dense(config.num_classes, activation='softmax')(encoded_dialogue)
    network = Model(inputs= input_layer, outputs= output_layer)

    if(print_network):
        network.summary()

    return network

## metric monitor while training, not used so far
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

def train(input_dir, output_dir):

    #load raw data
    data = {}
    with utils.timer('load raw data'):
        for m in ['train', 'dev', 'test']:
            data[m] = {'text': [], 'target': []}
            with open('%s/%s.txt' % (input_dir, m), 'r') as i_file:
                for line in i_file:
                    parts = line.rstrip().split('^')
                    data[m]['target'].append(parts[0])
                    data[m]['text'].append('^'.join(parts[1:]))
            print('%s size %s' % (m, len(data[m]['target'])))

    _print_memory_usage()

    # text preprocess
    cnt = 500
    X = {}
    y = {}
    word_set = set()
    with utils.timer('text preprocess'):
        for m in ['train', 'dev', 'test']:
            X[m] = []
            y[m] = []
            for i in range(len(data[m]['text'])):
                if((config.debug == True) & (i == cnt)): # debug
                    break
                text = data[m]['text'][i]
                target = data[m]['target'][i]

                # one-hot encode for categorical target
                target_encoded = np.zeros(config.num_classes, dtype= np.int32)
                target_encoded[config.targets_encode[target]] = 1
                y[m].append(target_encoded)

                # clean text with nltk toolkit
                word_vectors = [word_tokenize(sentence) for sentence in text.rstrip().rsplit('^')]
                # collect words
                for sent in word_vectors:
                    word_set.update(sent)

                # padding on sentences
                word_vectors = np.array([sent[:config.max_word] if(len(sent) >= config.max_word) else ([config.default_word] * (config.max_word - len(sent))) + sent for sent in word_vectors])

                # padding on dialogues
                if(len(word_vectors) >= config.max_sentence):
                    word_vectors = word_vectors[:config.max_sentence,:]
                else:
                    word_vectors = np.concatenate((np.full((config.max_sentence - len(word_vectors), config.max_word), config.default_word), word_vectors), axis= 0)

                X[m].append(word_vectors)

                if(i % 200 == 0):
                    print('%s done.' % i)

                # garbage collection
                del word_vectors
                gc.collect()

            X[m] = np.array(X[m])
            y[m] = np.array(y[m])

            print('%s done.' % m)
            print(X[m].shape)

            _print_memory_usage()
    word_set.add(config.default_word)

    print(X['train'].shape)
    print(y['train'].shape)

    # load word embedding vectors
    with utils.timer('load word embedding vector'):
        emb_file = '%s/.keras/word2vec/glove.840B.300d.txt' % os.getenv('HOME')
        emb_dict = load_word_embedding_vectors(emb_file, word_set)
    print('embedding size: %s' % len(emb_dict))

    _print_memory_usage()

    # word to vector
    with utils.timer('word2vec'):
        for m in ['train', 'dev', 'test']:
            X[m] = np.array([[[emb_dict.get(w, emb_dict['unk']) for w in sent] for sent in dialogue] for dialogue in X[m]])
            print('%s done.' % m)

    print(X['train'].shape)
    print(y['train'].shape)
    print(np.unique(y['train']))

    # garbage collection
    del emb_dict
    gc.collect()

    # model compilation
    model = get_network(sentence_dim= config.sentence_dim, dialogue_dim= config.dialogue_dim, print_network= True)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    # early stopping
    early_stopping = EarlyStopping(monitor= 'val_acc', mode='max', patience= 20, verbose= 2)

    # train
    with utils.timer('fitting'):
        model.fit(X['train'], y['train'],
                batch_size= config.batch_size,
                epochs= config.epochs,
                validation_data= (X['dev'], y['dev']),
                callbacks= [early_stopping],
                verbose= 2)

    _print_memory_usage()

    # test
    loss, acc = model.evaluate(X['test'], y['test'])
    print(loss)
    print(acc)

    # save model
    ## TODO

if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()

    parser.add_argument('-pipeline', '--pipeline',
                        default= 'model',
                        choices= ['preprocess', 'model'])

    parser.add_argument('-strategy', "--strategy",
                        default= 'mtqa_rnn',
                        help= "strategy",
                        choices= ['mtqa_rnn'])

    parser.add_argument('-phase', "--phase",
                        default= 'train',
                        help= "project phase",
                        choices= ['format', 'train', 'export', 'test'])

    parser.add_argument('-data_input', '--data_input',
                        default= '%s/raw' % (config.database_dir)
                        )
    args = parser.parse_args()

    if(args.pipeline == 'preprocess'):
        if(args.phase == 'format'):
            data_utils.format_raw_data(args.data_input, source= 'kvret')
    elif(args.pipeline == 'model'):
        if(args.phase == 'train'):
            raw_data_input = '%s/DriverAndAssistant/format' % args.data_input
            model_output_dir = '%s/%s' % (config.model_root_dir, args.strategy)
            if(os.path.exists(model_output_dir) == False):
                os.makedirs(model_output_dir)
            train(raw_data_input, model_output_dir)
