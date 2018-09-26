# configurations of this project
#
# Created by yuanpingzhou at 9/25/18

database_dir = '../data'
model_root_dir = '%s/model' % database_dir
default_word = 'unk' # used within glove
debug = False # turn it on while in debug mode, otherwise it will be in production mode

# target code map
targets_encode = {'navigate': 0, 'schedule': 1, 'weather': 2}

# model arch params
num_classes = len(targets_encode) # number of object classes
sentence_dim = 128# number of rnn states within sentence level, namely sentence vector length
dialogue_dim = 64# number of rnn states within dialogue level, namely dialogue vector length

max_sentence = 10 # maximum of sentences in a dialogue/session
max_word = 50 # maximum of words in a sentence
word_embedding_dim = 300 # dimension of word embedding, fixed in glove

# train  params
batch_size = 32 # batch size while training
epochs = 50 # maximum training epochs
