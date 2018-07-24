import array
import hashlib
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os,sys
import config
import utils
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *

cs_delete_file = '%s/raw/内容联系方式样本_0716.xlsx' % config.DataBaseDir
pos_58_file = '%s/raw/58_2d_55-85_positive_labeled.csv' % config.DataBaseDir
neg_58_file = '%s/raw/58_2d_25-45_negative_labeled.csv' % config.DataBaseDir

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

# Hashed n-grams with 1 < n <= N_GRAM are included as features
# in addition to unigrams.
N_GRAM = 2

# Size of vocabulary; less frequent words will be treated as "unknown"
VOCAB_SIZE = 100000

# Number of buckets used for hashing n-grams
N_BUCKETS = 1000000

# Size of the embedding vectors
EMBEDDING_SIZE = 50

# Number of epochs for which the model is trained
N_EPOCH = 5

# Size of training mini-batches
BATCH_SIZE = 32

# Path to which to save the trained model
MODEL_FILE_PATH = 'model.npz'


class FastTextClassifier(object):
    """Simple wrapper class for creating the graph of FastText classifier."""

    def __init__(self, vocab_size, embedding_size, n_labels):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_labels = n_labels

        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')

        # Network structure
        network = AverageEmbeddingInputlayer(self.inputs, self.vocab_size, self.embedding_size)
        self.network = DenseLayer(network, self.n_labels)

        # Training operation
        cost = tl.cost.cross_entropy(self.network.outputs, self.labels, name='cost')
        self.train_op = tf.train.AdamOptimizer().minimize(cost)

        # Predictions
        self.prediction_probs = tf.nn.softmax(self.network.outputs)
        self.predictions = tf.argmax(self.network.outputs, axis=1, output_type=tf.int32)
        # self.predictions = tf.cast(tf.argmax(             # for TF < 1.2
        #     self.network.outputs, axis=1), tf.int32)

        # Evaluation
        are_predictions_correct = tf.equal(self.predictions, self.labels)
        self.auc = tf.metrics.auc(self.labels, self.prediction_probs[:,1])
        self.precision = tf.metrics.precision(self.labels, self.predictions)
        self.recall = tf.metrics.recall(self.labels, self.predictions)
        #self.accuracy = tf.reduce_mean(tf.cast(are_predictions_correct, tf.float32))

    def save(self, sess, filename):
        tl.files.save_npz(self.network.all_params, name=filename, sess=sess)

    def load(self, sess, filename):
        tl.files.load_and_assign_npz(sess, name=filename, network=self.network)


def augment_with_ngrams(unigrams, unigram_vocab_size, n_buckets, n=2):
    """Augment unigram features with hashed n-gram features."""

    def get_ngrams(n):
        return list(zip(*[unigrams[i:] for i in range(n)]))

    def hash_ngram(ngram):
        bytes_ = array.array('L', ngram).tobytes()
        hash_ = int(hashlib.sha256(bytes_).hexdigest(), 16)
        return unigram_vocab_size + hash_ % n_buckets

    return unigrams + [hash_ngram(ngram) for i in range(2, n + 1) for ngram in get_ngrams(i)]

def train_test_and_save_model():
    ## load data
    with utils.timer('Load data'):
        data_1 = utils.load_cs_deleted_data(cs_delete_file)
        print('target ratio: ')
        print(data_1['label'].value_counts())
        data_2 = utils.load_58_data(pos_58_file)
        print(data_2['label'].value_counts())
        data_3 = utils.load_58_data(neg_58_file)
        print(data_3['label'].value_counts())
        data = pd.concat([data_1, data_2, data_3], axis= 0, ignore_index= True)
        DebugDir = '%s/debug' % config.DataBaseDir
        if(os.path.exists(DebugDir) == False):
            os.makedirs(DebugDir)
        #writer = pd.ExcelWriter('%s/raw.xlsx' % DebugDir)
        #data.to_excel(writer, index= False)
        #writer.close()
        del data_3, data_2, data_1
        gc.collect()

    X_raw_words = data['text'].apply(utils.cut)
    uni_words = list(set([w for rec in X_raw_words for w in rec]))
    word_dict = dict(zip(uni_words, range(len(uni_words))))
    X_words = []
    for rec in X_raw_words:
        new_rec = []
        for w in rec:
            new_rec.append(word_dict[w])
        X_words.append(new_rec)
    # X_words = np.array(X_words)
    y = np.array(data['label'])
    if N_GRAM is not None:
        X_words = np.array([augment_with_ngrams(x, VOCAB_SIZE, N_BUCKETS, n= N_GRAM) for x in X_words])

    print(X_words.shape)
    print(y.shape)
    print(X_words[:5])
    print(y[:5])

    final_train_pred = np.zeros(len(X_words))
    for s in range(config.train_times):
        s_start = time.time()
        train_pred = np.zeros(len(X_words))

        classifier = FastTextClassifier(
            vocab_size=VOCAB_SIZE + N_BUCKETS,
            embedding_size=EMBEDDING_SIZE,
            n_labels=2,
        )

        skf = StratifiedKFold(config.kfold, random_state=2018 * s, shuffle=False)

        for fold, (train_index, valid_index) in enumerate(skf.split(X_words, y)):
            X_train, X_valid = X_words[train_index], X_words[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            with tf.Session() as sess:
                sess.run(tf.local_variables_initializer())
                tl.layers.initialize_global_variables(sess)

                for epoch in range(N_EPOCH):
                    start_time = time.time()
                    print('Epoch %d/%d' % (epoch + 1, N_EPOCH))
                    for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True):
                        sess.run(
                            classifier.train_op, feed_dict={
                                classifier.inputs: tl.prepro.pad_sequences(X_batch),
                                classifier.labels: y_batch,
                            }
                        )

                    valid_pred_proba = sess.run(
                        classifier.prediction_probs, feed_dict={
                            classifier.inputs: tl.prepro.pad_sequences(X_valid)
                        }
                    )[:,1]
                    valid_pred_label = utils.proba2label(valid_pred_proba)
                    valid_auc = roc_auc_score(y_valid, valid_pred_proba)
                    valid_precision = precision_score(y_valid, valid_pred_label)
                    valid_recall = recall_score(y_valid, valid_pred_label)
                    if(epoch == N_EPOCH - 1):
                        train_pred[valid_index] = valid_pred_proba

                    # valid_precision = sess.run(
                    #     classifier.precision, feed_dict={
                    #         classifier.inputs: tl.prepro.pad_sequences(X_valid),
                    #         classifier.labels: y_valid,
                    #     }
                    # )
                    # valid_recall = sess.run(
                    #     classifier.recall, feed_dict={
                    #         classifier.inputs: tl.prepro.pad_sequences(X_valid),
                    #         classifier.labels: y_valid,
                    #     }
                    # )
                    print('valid: auc %.6f, precision %.6f, recall %.6f, took %s[s]' % (valid_auc, valid_precision, valid_recall, int(time.time() - start_time)))
                classifier.save(sess, MODEL_FILE_PATH)
            print('fold %s done!!!' % fold)
        auc = roc_auc_score(y, train_pred)
        precision = precision_score(y, utils.proba2label(train_pred))
        recall = recall_score(y, utils.proba2label(train_pred))
        print('auc %.6f, precision %.6f, recall %.6f, took %s[s]' % (auc, precision, recall, int(time.time() - s_start)))

if __name__ == '__main__':
    train_test_and_save_model()
