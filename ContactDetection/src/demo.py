import utils
import config
import tensorlayer as tl
import os,sys
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import xgboost

pd.set_option('display.max_rows', None)

cs_delete_file = '%s/raw/内容联系方式样本_0716.xlsx' % config.DataBaseDir
pos_58_file = '%s/raw/58_2d_55-85_positive_labeled.csv' % config.DataBaseDir
neg_58_file = '%s/raw/58_2d_25-45_negative_labeled.csv' % config.DataBaseDir

num_round = 1000
param = {
    #'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.2,
    'max_depth': 8,
    'silent': 1,
    'nthread': 4,
    'colsample_bytree': .4,
    'subsample': .9,
}

if __name__ == '__main__':
    ''''''
    ## load word2vec lookup table
    with utils.timer('Load word vector'):
        word2vec = tl.files.load_npy_to_any(name= '%s/model/word2vec_post_text_3d.npy' % config.DataBaseDir)

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
        writer = pd.ExcelWriter('%s/raw.xlsx' % DebugDir)
        data.to_excel(writer, index= False)
        writer.close()
        del data_3, data_2, data_1
        gc.collect()

    ## representation
    hit_words = []
    with utils.timer('representation'):
        X_words = data['text'].apply(utils.word_seg)
        X_words = np.array([wstr.split(' ') for wstr in X_words])
        y = data['label']
        X = np.zeros((X_words.shape[0], config.embedding_size), dtype= 'float32')
        for x_idx in range(X_words.shape[0]):
            word_set = set([w for w in X_words[x_idx] if(w in word2vec)])
            hit_words.append(list(word_set))
            row_vec = [word2vec[w] for w in list(word_set)]
            if(len(row_vec) > 0):
                X[x_idx] = np.array(row_vec).mean(axis= 0)
        #del X_words
        #gc.collect()

    #X_train_valid = X[:-int(X.shape[0] * config.test_ratio),]
    #y_train_valid = y[:-int(X.shape[0] * config.test_ratio)]
    X_train_valid = X.copy()
    y_train_valid = y.copy()

    assert (np.sum(data['label'] == np.array(y_train_valid)) == len(data))
    #X_test = X[-int(X.shape[0] * config.test_ratio):,]
    #y_test = y[-int(X.shape[0] * config.test_ratio):]
    del X, y
    gc.collect()

    final_train_pred = np.zeros(len(X_train_valid))
    #final_test_pred = np.zeros(len(X_test))
    for s in range(config.train_times):
        train_pred = np.zeros(len(X_train_valid))
        #test_pred = np.zeros(len(X_test))
        skf = StratifiedKFold(config.kfold, random_state= 2018 * s, shuffle= False)
        for fold, (train_index, valid_index) in enumerate(skf.split(X_train_valid, y_train_valid)):
            X_train, X_valid = X_train_valid[train_index,:], X_train_valid[valid_index,:]
            y_train, y_valid = y_train_valid[train_index], y_train_valid[valid_index]

            # LR
            model = LogisticRegression(C= 100.0)
            model.fit(X_train, y_train)
            ## infer
            valid_pred_proba = model.predict_proba(X_valid)[:,1]
            valid_pred_label = model.predict(X_valid)
            #test_pred_proba = model.predict_proba(X_test)[:,1]

            ## XGB
            # xg_train = xgboost.DMatrix(X_train, label=y_train)
            # xg_valid = xgboost.DMatrix(X_valid, label=y_valid)
            # watchlist = [(xg_train, 'train'), (xg_valid, 'valid')]
            # model = xgboost.train(param, xg_train, num_round, watchlist, early_stopping_rounds=100,verbose_eval= 50)
            # ## infer
            # valid_pred_proba = model.predict(xg_valid)
            # test_pred_proba = model.predict(xgboost.DMatrix(X_test))
            # valid_pred_label = utils.proba2label(valid_pred_proba)

            ## evaluate
            auc = roc_auc_score(y_valid, valid_pred_proba)
            precision = precision_score(y_valid, valid_pred_label)
            recall = recall_score(y_valid, valid_pred_label)
            ##
            train_pred[valid_index] = valid_pred_proba
            #test_pred += test_pred_proba
            ##
            print('fold %s: auc %.6f, precision %.6f, recall %.6f' % (fold, auc, precision, recall))
            print(np.sum(valid_pred_label), np.sum(y_valid))
        ##
        #test_pred /= config.kfold
        train_pred_label = [1 if(v > 0.5) else 0 for v in train_pred]
        t_auc = roc_auc_score(y_train_valid, train_pred)
        t_precision = precision_score(y_train_valid, train_pred_label)
        t_recall = recall_score(y_train_valid, train_pred_label)
        print('\n---------------------------')
        print('#%s: auc %.6f, precision %.6f, recall %.6f' % (s, t_auc, t_precision, t_recall))
        print(np.sum(train_pred_label), np.sum(y_train_valid))
        print('---------------------------\n')

        ## DEBUG
        DebugDir = '%s/debug' % config.DataBaseDir
        if(os.path.exists(DebugDir) == False):
            os.makedirs(DebugDir)
        writer = pd.ExcelWriter('%s/lr_bad_case.xlsx' % DebugDir)
        error_indexs = [i for i in range(len(y_train_valid)) if(train_pred_label[i] != y_train_valid[i])]
        with open('%s/text.txt' % DebugDir, 'w') as text_file, \
                open('%s/words.txt' % DebugDir, 'w') as words_file, \
                open('%s/result.txt' % DebugDir, 'w') as result_file:
            for i in error_indexs:
                text_file.write('%s\n' % data['text'][i])
                words_file.write('%s\n' % (' '.join(X_words[i])))
                result_file.write('%s|%s|%s\n' % (' '.join(hit_words[i]), y_train_valid[i], train_pred_label[i]))
        text_file.close()
        words_file.close()
        # debug_df = pd.DataFrame()
        # debug_df['id'] = error_indexs
        # debug_df['text'] = [v.replace('\n', ' ') for v in data['text'][error_indexs]]
        # # debug_df['text'] = data['text'][error_indexs]
        # debug_df['words'] = [' '.join(ws).replace('\n', ' ') for ws in X_words[error_indexs]]
        # debug_df['label'] = y_train_valid[error_indexs]
        # debug_df['predict_proba'] = train_pred[error_indexs]
        # debug_df['predict_label'] = np.array(train_pred_label)[error_indexs]
        # debug_df.to_excel(writer, sheet_name= 'bad case for lr', index= False)
        # writer.save()
        # for i in error_indexs[:200]:
        #     print(data['text'][i], data['label'][i], y_train_valid[i], train_pred_label[i], i)

        ##
        final_train_pred += train_pred
        #final_test_pred += test_pred
        final_train_pred_label = [1 if(v > 0.5) else 0 for v in final_train_pred/(s + 1)]
        agg_auc = roc_auc_score(y_train_valid, final_train_pred/(s + 1))
        agg_precision = precision_score(y_train_valid, final_train_pred_label)
        agg_recall = recall_score(y_train_valid, final_train_pred_label)
        print('\n---------------------------')
        print('#%s[agg]: auc %.6f, precision %.6f, recall %.6f' % (s, agg_auc, agg_precision, agg_recall))
        print('---------------------------\n')
