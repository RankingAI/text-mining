import logging
import math
import os
import random
import sys
import shutil
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorflow import saved_model
from tensorflow.python.util import compat

from sklearn.model_selection import train_test_split

import config
import utils
import pandas as pd

import gc, datetime

cs_delete_file_1 = '%s/raw/内容联系方式样本_0716.xlsx' % config.DataBaseDir
cs_delete_file_2 = '%s/raw/内容联系方式样本_0725.xlsx' % config.DataBaseDir
pos_58_file = '%s/raw/58_2d_55-85_positive_labeled.csv' % config.DataBaseDir
neg_58_file = '%s/raw/58_2d_25-45_negative_labeled.csv' % config.DataBaseDir
test_file = '%s/raw/test.txt' % config.DataBaseDir

resume = False
model_name = 'drnn'
model_version = 5
#datestr = datetime.datetime.now().strftime("%Y%m%d")
datestr = '20180809'
OutputDir = '%s/%s' % (config.ModelOutputDir, model_name)
if(os.path.exists(OutputDir) == False):
    os.makedirs(OutputDir)

def load_dataset(test_size=0.2):
    ## load word vector
    with utils.timer('Load word vector'):
        word2vec = tl.files.load_npy_to_any(name='%s/word2vec/w2v_sgns_%s_%s_%s.npy' % (config.ModelOutputDir, config.embedding_size, config.corpus_version, datestr))
    ## load train data
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
        data = pd.concat([data_1, data_2, data_3, data_4[data_4['label'] == 1].reset_index(drop=True)], axis=0,ignore_index=True)
        #data = pd.concat([data_1, data_2, data_3, data_4], axis=0,ignore_index=True)
        DebugDir = '%s/debug' % config.DataBaseDir
        if (os.path.exists(DebugDir) == False):
            os.makedirs(DebugDir)
        del data_4, data_3, data_2, data_1
        gc.collect()
    ## data representation
    with utils.timer('representation for train'):
        # X = [[word2vec.get(w, word2vec['_UNK']) for w in utils.cut(text)] for text in data['text'].values]
        X = []
        y = []
        for i in range(len(data)):
            text = data['text'][i]
            if(text == ''):
                continue
            words = utils.cut(text)
            if(len(words) == 0):
                continue
            X.append([word2vec.get(w, word2vec['_UNK']) for w in words])
            y.append(data['label'][i])

    del word2vec, data
    gc.collect()

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size)

    return X_train, y_train, X_valid, y_valid

def load_test():
    ## load word vector
    with utils.timer('Load word vector'):
        word2vec = tl.files.load_npy_to_any(name='%s/word2vec/w2v_sgns_%s_%s_%s.npy' % (
        config.ModelOutputDir, config.embedding_size, config.corpus_version, datestr))
    ## load train data
    with utils.timer('Load test data'):
        test_data, uid_list, info_id_list = utils.load_test_data(test_file)
        test_data, uid_list, info_id_list  = test_data[:int(0.2 * len(test_data))], uid_list[:int(0.2 * len(uid_list))], info_id_list[:int(0.2 * len(info_id_list))]
    with utils.timer('representation for test'):
        X_test = []
        text_test = []
        for i in range(len(test_data)):
            text = test_data[i]
            if(text == ''):
                continue
            words = utils.cut(text)
            if(len(words) == 0):
                continue
            X_test.append([word2vec.get(w, word2vec['_UNK']) for w in words])
            text_test.append(text)
    del word2vec
    gc.collect()

    return X_test, text_test, uid_list, info_id_list

def network(x, keep= 0.75, is_train= True):
    n_hidden = 128 # hidden layer num of features
    input = tl.layers.InputLayer(x, name='input_layer')
    #input = tl.layers.DropoutLayer(input, keep= 0.5, is_train= is_train)
    rnn = tl.layers.DynamicRNNLayer(input,
        cell_fn         = tf.nn.rnn_cell.LSTMCell,
        n_hidden        = n_hidden,
        dropout         = keep,
        sequence_length = tl.layers.retrieve_seq_length_op(x),
        return_seq_2d   = True,
        return_last     = True,
        name            = 'dynamic_rnn')
    #max_rnn = tl.layers.GlobalMaxPool1d(rnn, name= 'max_pool')
    #max_rnn = tl.layers.FlattenLayer(max_rnn, name= 'flatten_max_rnn')
    #avg_rnn = tl.layers.GlobalMeanPool1d(rnn, name= 'avg_pool')
    #avg_rnn = tl.layers.FlattenLayer(avg_rnn, name= 'flatten_avg_rnn')
    #conc = tl.layers.ConcatLayer([max_rnn, avg_rnn], concat_dim= 1, name= 'concate')

    #network = tl.layers.DenseLayer(conc, n_units= 1, act= tf.nn.sigmoid, name= 'output')
    #network.outputs_label = tf.cast(tf.greater_equal(network.outputs, 0.5), tf.int64)

    network = tl.layers.DenseLayer(rnn, n_units= 2,act=tf.identity, name="output")
    network.outputs_proba = tf.nn.softmax(network.outputs)[:,1]
    network.outputs_label = tf.argmax(tf.nn.softmax(network.outputs), 1)

    network.print_layers()
    return network

def load_checkpoint(sess, ckpt_file):
    ckpt = ckpt_file + '.ckpt'
    index = ckpt + ".index"
    meta  = ckpt + ".meta"
    if os.path.isfile(index) and os.path.isfile(meta):
        tf.train.Saver().restore(sess, ckpt)

def save_checkpoint(sess, ckpt_file):
    path = os.path.dirname(os.path.abspath(ckpt_file))
    if os.path.isdir(path) == False:
        logging.warning('Path (%s) not exists, making directories...', path)
        os.makedirs(path)
    tf.train.Saver().save(sess, ckpt_file)

def train(sess, x, network, ckpt_dir):
    lr = 0.005
    y         = tf.placeholder(tf.int64, [None, ], name="labels")
    cost      = tl.cost.cross_entropy(network.outputs, y, 'xentropy')
    #cost = tl.cost.binary_cross_entropy(network.outputs, tf.cast(y, tf.float32), name= 'binary_cross_entropy')
    optimizer = tf.train.AdamOptimizer(learning_rate= lr).minimize(cost)
    correct   = tf.equal(network.outputs_label, y)
    accuracy  = tf.reduce_mean(tf.cast(correct, tf.float32))

    # 使用TensorBoard可视化loss与准确率：`tensorboard --logdir=./logs`
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    writter_train = tf.summary.FileWriter('%s/logs/train' % OutputDir, sess.graph)
    writter_test  = tf.summary.FileWriter('%s/logs/test' % OutputDir)

    x_train, y_train, x_test, y_test = load_dataset(test_size= 0.2)
    logging.info("train size %s, test %s" % (len(x_train), len(x_test)))

    sess.run(tf.global_variables_initializer())
    if(resume):
        load_checkpoint(sess, '%s/%s' % (ckpt_dir, model_name))

    n_epoch      = 4
    batch_size   = 512
    test_size    = 1024
    display_step = 10
    step         = 0
    total_step   = math.ceil(len(x_train) / batch_size) * n_epoch
    logging.info("batch_size: %d", batch_size)
    logging.info("Start training the network...")
    for epoch in range(n_epoch):
        for batch_x, batch_y in tl.iterate.minibatches(x_train, y_train, batch_size, shuffle=True):
            start_time = time.time()
            max_seq_len = max([len(d) for d in batch_x])
            for i,d in enumerate(batch_x):
                batch_x[i] += [np.zeros(config.embedding_size) for i in range(max_seq_len - len(d))]
            batch_x = list(batch_x) # ValueError: setting an array element with a sequence.

            feed_dict = {x: batch_x, y: batch_y}
            sess.run(optimizer, feed_dict)

            # TensorBoard打点
            summary = sess.run(merged, feed_dict)
            writter_train.add_summary(summary, step)

            print('batch train[%s] done...\n' % len(batch_x))

            # 计算测试集准确率
            start = random.randint(0, len(x_test)-test_size)
            test_data  = x_test[start:(start+test_size)]
            test_label = y_test[start:(start+test_size)]
            max_seq_len = max([len(d) for d in test_data])
            for i,d in enumerate(test_data):
                test_data[i] += [np.zeros(config.embedding_size) for i in range(max_seq_len - len(d))]
            test_data = list(test_data)
            summary = sess.run(merged, {x: test_data, y: test_label})
            writter_test.add_summary(summary, step)

            print('batch inferring on test[%s] done...\n' % len(test_data))

            # 每十步输出loss值与准确率
            if ((step == 0) | ((step + 1) % display_step == 0) | (step == total_step - 1)):
                logging.info("Epoch %d/%d Step %d/%d took %fs", epoch + 1, n_epoch,
                             step + 1, total_step, time.time() - start_time)
                loss = sess.run(cost, feed_dict=feed_dict)
                acc  = sess.run(accuracy, feed_dict=feed_dict)
                logging.info("Minibatch Loss= " + "{:.6f}".format(loss) +
                             ", Training Accuracy= " + "{:.5f}".format(acc))
                save_checkpoint(sess, '%s/%s' % (ckpt_dir, model_name))
            step += 1

    max_seq_len = max([len(d) for d in x_test])
    for i,d in enumerate(x_test):
        x_test[i] += [np.zeros(config.embedding_size) for i in range(max_seq_len - len(d))]
    x_test = list(x_test) # ValueError: setting an array element with a sequence.

    acc_valid = sess.run(accuracy, {x: x_test, y: y_test})
    print('accuracy on whole valid %.4f' % acc_valid)

def export(model_version, model_dir, sess, x, y_op):
    if model_version <= 0:
        logging.warning('Please specify a positive value for version number.')
        sys.exit()

    path = os.path.dirname(os.path.abspath(model_dir))
    if os.path.isdir(path) == False:
        logging.warning('Path (%s) not exists, making directories...', path)
        os.makedirs(path)

    export_path = os.path.join(
        compat.as_bytes(model_dir),
        compat.as_bytes(str(model_version)))

    if os.path.isdir(export_path) == True:
        logging.warning('Path (%s) exists, removing directories...', export_path)
        shutil.rmtree(export_path)

    builder = saved_model.builder.SavedModelBuilder(export_path)
    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y_op)

    prediction_signature = saved_model.signature_def_utils.build_signature_def(
        inputs={'x': tensor_info_x},
        outputs={'y': tensor_info_y},
        method_name= saved_model.signature_constants.PREDICT_METHOD_NAME)

    builder.add_meta_graph_and_variables(
        sess,
        [saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_text': prediction_signature,
            saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
        })

    builder.save()

if __name__ == '__main__':
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    ckpt_dir = '%s/checkpoint' % OutputDir
    if(os.path.exists(ckpt_dir) == False):
        os.makedirs(ckpt_dir)

    x = tf.placeholder("float", [None, None, config.embedding_size], name="inputs")
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction= 0.7)))

    flags = tf.flags
    flags.DEFINE_string("mode", "test", "train or export or test")
    FLAGS = flags.FLAGS

    if FLAGS.mode == "train":
        network = network(x)
        train(sess, x, network, ckpt_dir)
        logging.info("Optimization Finished!")
    elif FLAGS.mode == "export":
        model_dir    = '%s/infer' % OutputDir
        network = network(x, keep=1.0, is_train= False)
        sess.run(tf.global_variables_initializer())
        load_checkpoint(sess, ckpt_dir)
        export(model_version, model_dir, sess, x, network.outputs_label)
        logging.info("Servable Export Finishied!")
    elif FLAGS.mode == 'test':
        X_test, text_test, uid_list, info_id_list = load_test()
        print('test size %s' % len(X_test))

        network = network(x, keep= 1.0, is_train= False)
        sess.run(tf.global_variables_initializer())
        load_checkpoint(sess, ckpt_dir)

        max_seq_len = max([len(d) for d in X_test])
        for i, d in enumerate(X_test):
            X_test[i] += [np.zeros(config.embedding_size) for i in range(max_seq_len - len(d))]
        x_test = list(X_test)  # ValueError: setting an array element with a sequence.
        y_pred = sess.run(network.outputs_label, {x: X_test})
        text_test = [text.replace(',', ' ') for text in text_test]

        test_dir = '%s/test' % OutputDir
        if(os.path.exists(test_dir) == False):
            os.mkdir(test_dir)

        outdf = pd.DataFrame({'text': text_test, 'uid': uid_list, 'info_id': info_id_list, 'predict_label': y_pred})
        outdf[['uid', 'info_id', 'text', 'predict_label']].to_csv('%s/%s_%s.csv' % (test_dir, model_name, datestr), header= True, index= False, encoding= 'utf-8')
