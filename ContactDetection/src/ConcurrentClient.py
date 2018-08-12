from threading import Thread
import numpy as np
import jieba
import tensorlayer as tl
from grpc.beta import implementations
import utils
import config
import tensorflow as tf
import time

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

server_ip = '192.168.0.112'
server_port= 8500
#server_ip = '10.135.9.2'
#server_port = 8502
num_threads = 10
timeout = 5
results = np.full(num_threads, -1.0)

class TextRequest(Thread):
    def __init__(self, no, text):
        super().__init__()
        ''''''
        self._no = no
        self._text = text
        self.channel = implementations.insecure_channel(server_ip, server_port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = 'saved_model.pb'
        self.request.model_spec.signature_name = 'predict'

    def run(self):
        ''''''
        for i in range(1):
            s1 = time.time()
            send_text = text_tensor(self._text)
            s2 = time.time()
            print('pre-processing %s' % (s2 - s1))
            self.request.inputs['inputs'].ParseFromString(tf.contrib.util.make_tensor_proto(send_text, dtype=tf.float32).SerializeToString())
            s3 = time.time()
            print('parse request %s' % (s3 - s2))
            # request.inputs['inputs'].CopyFrom(tf.contrib.util.make_tensor_proto(send_text, dtype=tf.float32))
            try:
                ret_proto = self.stub.Predict(self.request, timeout).outputs['output']
                s4 = time.time()
                print('send %s' % (s4 - s3))
                ret_list = tf.contrib.util.make_ndarray(ret_proto)
                results[self._no] = ret_list[0][0]
                print(ret_list)
                s5 = time.time()
                print('parse response %s' % (s5 - s4))
            except:
               return

def text_tensor(text):
    words = [w for w in utils.cut(text)]
    print(words)
    if(len(words) < 150):
        words = ['_UNK'] * (150 - len(words)) + words
    else:
        words = words[:150]
    words = [word2vec.get(w, word2vec['_UNK']) for w in words]
    words = np.asarray(words)

    sample = words.reshape(1, len(words), config.embedding_size)
    return sample

sample_text = '洗锅神器 需要的可以私聊我vliubo185 免费送100份 一份3个 全新品质'
#sample_text = '王者荣耀陪练 教授技术，走位，意识，铭文搭配，出装  全局技术指导。'
jieba.lcut(sample_text, cut_all= False)

with utils.timer('Load word vector'):
    word2vec = tl.files.load_npy_to_any(name='%s/word2vec/w2v_sgns_500_post_text_7d_20180803.npy' % config.ModelOutputDir)

threads = [TextRequest(i, sample_text) for i in range(num_threads)]
start = time.time()
with utils.timer('%s REQUEST' % num_threads):
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(results)
    print('success %s/%s ' % (len(np.where(results > 0.5)[0]), num_threads))
end = time.time()
print('time %s' % (end - start))
