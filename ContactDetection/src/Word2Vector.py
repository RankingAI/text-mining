import utils
import config
import logging, os, sys
import tensorflow as tf
import tensorlayer as tl
import collections
import gc
import time, datetime
import numpy as np
import json

tf.logging.set_verbosity(tf.logging.DEBUG)

debug = True
corpus_file = '%s/raw/%s.txt' % (config.DataBaseDir, config.corpus_version)
OutputDir = '%s/word2vec' % config.ModelOutputDir
model_name = 'w2v_sgns' # word2vec: skip gram with negative sampling
datestr = datetime.datetime.now().strftime("%Y%m%d")

if(os.path.exists(OutputDir) == False):
    os.makedirs(OutputDir)

g_params = {
    'min_freq': 20,
    'batch_size': 1200,
    'embedding_size': config.embedding_size,
    'skip_window': 3,
    'num_sampled': 128,
    'learning_rate': 0.8,
    'n_epoch': 20,
    'minority_replacement': '_UNK',
    'verbose': 500,
    'resume': False,
    'strategy': '%s_%s_%s_%s' % (model_name, config.embedding_size, config.corpus_version, datestr)
}

class Word2Vec:
    ''''''
    def __init__(self, data, params):
        ''''''
        self.model_name = params['strategy']
        self.words = [w for rec in data for w in rec]
        ## debug
        #print('wx count %s' % np.sum([1 for w in self.words if((w == 'wx'))]))
        ##
        self.vocabulary_size = len([w for w, c in collections.Counter(self.words).most_common() if(c > params['min_freq'])])
        self.batch_size = params['batch_size']
        self.embedding_size = params['embedding_size']
        self.skip_window = params['skip_window']
        self.num_skips = 2 * self.skip_window
        self.num_sampled = params['num_sampled']
        self.learning_rate = params['learning_rate']
        self.n_epoch = params['n_epoch']
        self.verbose = params['verbose']
        self.resume = params['resume']
        self.num_steps = int((len(self.words) / self.batch_size) * self.n_epoch)  # total number of iteration
        self.minority_replacement = params['minority_replacement']
        ## GPU configuration
        self.tf_sess = tf.Session(config=tf.ConfigProto(gpu_options= tf.GPUOptions(per_process_gpu_memory_fraction= 0.7)))

    def __load_checkpoint(self, ckpt_file_path):
        ckpt = ckpt_file_path + '.ckpt'
        index = ckpt + ".index"
        meta = ckpt + ".meta"
        if os.path.isfile(index) and os.path.isfile(meta):
            tf.train.Saver().restore(self.tf_sess, ckpt)

    def __save_checkpint(self, ckpt_file_path):
        ''''''
        path = os.path.dirname(os.path.abspath(ckpt_file_path))
        if os.path.isdir(path) == False:
            logging.warning('Path (%s) not exists, making directories...', path)
            os.makedirs(path)
        tf.train.Saver().save(self.tf_sess, ckpt_file_path + '.ckpt')

    def train(self):
        ''''''
        ## build words dataset
        data, count, dictionary, reverse_dictionary = tl.nlp.build_words_dataset(self.words,self.vocabulary_size,True,self.minority_replacement)
        # save vocabulary to txt
        tl.nlp.save_vocab(count, name='%s/vocabulary_%s.txt' % (OutputDir, self.model_name))
        del self.words
        gc.collect()
        print('-------------------------------------------')
        print('Most 5 common words (+UNK)', count[:5])
        print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
        print('-------------------------------------------')
        ## embedding network
        train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        # Look up embeddings for inputs.
        emb_net = tl.layers.Word2vecEmbeddingInputlayer(
            inputs= train_inputs,
            train_labels= train_labels,
            vocabulary_size= self.vocabulary_size,
            embedding_size= self.embedding_size,
            num_sampled= self.num_sampled,
            nce_loss_args= {},
            E_init= tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
            E_init_args= {},
            nce_W_init= tf.truncated_normal_initializer(stddev=float(1.0 / np.sqrt(self.embedding_size))),
            nce_W_init_args= {},
            nce_b_init= tf.constant_initializer(value=0.0),
            nce_b_init_args= {},
            name= 'word2vec_layer',
        )
        cost = emb_net.nce_cost
        train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)

        ## initialize global weights
        tl.layers.initialize_global_variables(self.tf_sess)
        ckpt_file_path = "%s/checkpoint" % OutputDir
        if(os.path.exists(ckpt_file_path) == False):
            os.makedirs(ckpt_file_path)
        if self.resume:
            self.__load_checkpoint('%s/%s' % (ckpt_file_path, self.model_name))

        emb_net.print_params(False)
        emb_net.print_layers()

        data_index = 0
        loss_list = []
        for step in range(self.num_steps):
            start = time.time()
            batch_inputs, batch_labels, data_index = tl.nlp.generate_skip_gram_batch(data= data,
                                                                                     batch_size= self.batch_size,
                                                                                     num_skips= self.num_skips,
                                                                                     skip_window= self.skip_window,
                                                                                     data_index= data_index)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            # We perform one update step by evaluating the train_op (including it
            # in the list of returned values for sess.run()
            _, loss_val = self.tf_sess.run([train_op, cost], feed_dict=feed_dict)
            end = time.time()

            loss_list.append(loss_val)
            if((step % (self.verbose + 1) == 0) & (step != 0)):
                print('Average loss at step %d/%d, loss: %.6f, take %ss' % (step, self.num_steps, np.mean(loss_list), int(end - start)))

            ## saving
            if(((step < self.num_steps - 1) & (step % ((self.verbose + 1) * 10) == 0)) | (step == self.num_steps - 1)):
                print("#%s, Save model, data and dictionaries %s" % (step, "!" * 10))
                ## Save to ckpt file
                self.__save_checkpint('%s/%s' % (ckpt_file_path, self.model_name))

                ## saving embedding matrix
                embedding_file_path = "%s/%s" % (OutputDir, self.model_name)
                uni_words = list(dictionary.keys())
                uni_ids = list(dictionary.values())
                embedding_matrix = self.tf_sess.run(tf.nn.embedding_lookup(emb_net.normalized_embeddings, tf.constant(uni_ids, dtype=tf.int32)))
                word2emb = dict(zip(uni_words, embedding_matrix))
                tl.files.save_any_to_npy(save_dict= word2emb,name= embedding_file_path)
                with open('%s/%s.json' % (OutputDir, self.model_name), 'w') as o_file:
                    for k in word2emb:
                        o_file.write('%s\n' % json.dumps({k: word2emb[k].flatten().tolist()}, ensure_ascii=False))
                o_file.close()

if __name__ == '__main__':
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    ## load corpus
    with utils.timer('Loading corpus'):
        corpus = utils.load_corpus(corpus_file, debug)

    ## train word2vec with skip-gram & negative sampling
    with utils.timer('word2vec training'):
        Word2Vec(corpus, g_params).train()
