import pandas as pd
import jieba
import re, time, datetime
from contextlib import contextmanager
import os,sys
import numpy as np

## timer function
@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'\n[{name}] done in {time.time() - t0:.0f} s')

def load_cs_deleted_data(excel_file, sheet_name= 'Sheet1'):
    ''''''
    xl = pd.ExcelFile(excel_file)
    data = xl.parse(sheet_name)
    ## remove uselefinal_targetss columns
    data = data[['标题', '内容', '内容长度', '标志词', '原始联系方式', '翻译联系方式']]
    ## rename columns
    data.rename(index= str, columns= {'标题': 'title',
                                '内容': 'content',
                                '内容长度': 'content_len',
                                '标志词': 'keywords',
                                '原始联系方式': 'raw_target',
                                '翻译联系方式': 'final_target',
                               }, inplace= True)
    ## remove null rows
    none_null_indexs = data.index[(data['title'].isnull() != True) | (data['content'].isnull() != True)]
    print('none null rows %s' % len(none_null_indexs))
    data = data.loc[none_null_indexs,]
    ## convert type
    data['content_len'] = data['content_len'].astype('float32')
    data['title'] = data['title'].astype('str')
    data['content'] = data['content'].astype('str')
    data['text'] = data['title'] + ' ' + data['content']
    data['final_target'] = data['final_target'].astype('str')
    ## target
    data['label'] = 0
    pos_indexs = data.index[data['final_target'].str.replace(' ','') != '0']
    data.loc[pos_indexs, 'label'] = 1
    data.drop(['final_target', 'keywords', 'raw_target', 'title', 'content', 'content_len'], axis= 1, inplace= True)

    return data.reset_index(drop= True)

def load_58_data(txt_file, sheet_name= 'Sheet1'):
    ''''''
    n = 0
    text_list = []
    target_list = []
    err = 0
    with open(txt_file, 'r') as i_file:
        for line in i_file:
            line = line.strip()
            if(n == 0):
                n += 1
                continue
            if(line != None):
                try:
                    parts = line.split(',')
                    final_target = parts[-1].replace(' ', '')
                    raw_target = parts[-2].replace(' ', '')
                    wxscore = int(parts[-3].replace(' ',''))
                    text = ' '.join(parts[2:-3])
                    if((final_target != '') & (text != '')):
                        text_list.append(text)
                        if(final_target != '0'):
                            target_list.append(1)
                        else:
                            target_list.append(0)
                except:
                    err += 1
            n += 1
    i_file.close()
    data = pd.DataFrame()
    data['text'] = text_list
    data['label'] = target_list
    print('error %s' % err)
    return data

def load_corpus(text_file, debug= False):
    ''''''
    corpus = []
    n = 0
    with open(text_file, 'r') as i_file:
        for line in i_file:
            if(n == 0):
                n += 1
                continue
            if(line != None):
                if((debug == True) & (n == 80000)):
                    break
                    # corpus.append(word_seg(line).split(' '))
                corpus.append(word_seg(line).split(' '))
            n += 1
    i_file.close()

    return corpus

def load_test_data(text_file):
    n = 0
    data = []
    uid_list = []
    info_id_list = []
    with open(text_file, 'r') as i_file:
        for line in i_file:
            line = line.strip()
            if(n == 0):
                n += 1
                continue
            if((line != None) & (n % 20 == 0)):
                parts = line.split('\t')
                if((len(parts) >= 3) & parts[0].isnumeric() & parts[1].isnumeric()):
                    data.append('\t'.join(parts[2:]))
                    uid_list.append(parts[0])
                    info_id_list.append(parts[1])
            n += 1
    return data, uid_list, info_id_list

def is_chinese_words(data):
    ret = [w for w in data if ((w >= u'\u4e00') & (w <= u'\u9fa5'))]
    return len(ret) > 0

def cut(sentence):
    ''''''
    r_symbols = '[’!"#$%&\'()*,/:;<=>?@[\\]^`{|}~]+|[\t\n]+|[／“”　￼＝·《》～！，；【】：。？、~@#￥%……&*（）]+'
    r_float = "^\d+?\.\d+?$"
    r_wx = "^[a-z|A-Z]+[0-9]+$"
    ## replace symbols
    sentence = sentence.replace('+', '加')
    sentence = re.sub(r_symbols, ' ', sentence.strip())
    words = jieba.lcut(sentence, cut_all= False)
    words = [w for w in words if((w != '') & (w != ' '))] ## remove the blanks
    ##
    clean_words = []
    for w in words:
        if(w.isnumeric()): # integer
            clean_words.append('INTEGER_%s' % len(w))
        elif(re.match(r_float, w) != None): # float
            clean_words.append('FLOAT')
        elif((w.lower() == 'v') | (w.lower() == 'q') | (w.lower() == 'w')): # alpha
            clean_words.append(w.lower())
        elif((w.isalpha() == True) & (len(w) > 1) & (is_chinese_words(w) == False)):
            clean_words.append(w.lower())
        elif(is_chinese_words(w)):# & (len(w) > 1)) | (w == '加')): # chinese words
            clean_words.append(w)
        elif(re.match(r_wx, w) != None): # alpha + num
            if(len(w) <= 5):
                clean_words.append('ALNUM_LESS')
            elif('qq' not in w.lower()):
                clean_words.append('ALNUM_%s' % len(w))
            elif(w.lower().startswith('qq')):
                clean_words.append('qq')
                clean_words.append('INTEGER_%s' % (len(w) - 2))
        elif((w == '-') | (w == '_')):
            clean_words.append(w)
    return clean_words

def word_seg(sentence):
    return ' '.join(cut(sentence))

def proba2label(data):
    return [1 if(v > 0.5) else 0 for v in data]
