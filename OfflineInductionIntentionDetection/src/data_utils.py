# data related operation
#
# This QA dataset is is conversation between driver and car-assistant with thousands of sessions,
# more infomation you can refer https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/.
#
# Created by yuanpingzhou at 9/25/18

import json, os

def format_raw_data(data_input, source= 'kvret'):
    ''''''
    if(source == 'kvret'):
        raw_data_input = '%s/DriverAndAssistant' %  data_input
        raw_data_output = '%s/DriverAndAssistant/format' % data_input
        if(os.path.exists(raw_data_output) == False):
            os.makedirs(raw_data_output)
        for m in ['train', 'dev', 'test']:
            with open('%s/kvret_%s_public.json' % (raw_data_input, m), 'r') as i_file:
               json_data  = json.load(i_file)
            i_file.close()
            print('mode %s, data size %s' % (m, len(json_data)))
            dialogues = []
            targets = []
            for i in range(len(json_data)):
                dialogue = json_data[i]['dialogue']
                qa_pairs = []
                for j in range(len(dialogue)):
                    utterance = dialogue[j]['data']['utterance']
                    turn = dialogue[j]['turn']
                    end = dialogue[j]['data']['end_dialogue']
                    if((j == 0) & (turn == 'driver')): # the very first sentence is user
                        qa_pairs.append(utterance)
                    elif((j == 0) & (turn == 'assistant')):
                        pass
                    elif((turn == 'driver') & (turn == dialogue[j - 1]['turn'])): # last sentence is user self
                        qa_pairs.append(utterance)
                    elif((turn == 'driver') & (turn != dialogue[j - 1]['turn'])): # last sentence is agent
                        qa_pairs.append('%s %s' % (dialogue[j - 1]['data']['utterance'], utterance))
                    else:
                        pass
                dialogues.append('^'.join(qa_pairs))
                targets.append(json_data[i]['scenario']['task']['intent'])
            print(len(dialogues), len(targets))
            assert len(dialogues) == len(targets)
            with open('%s/%s.txt' % (raw_data_output, m), 'w') as o_file:
                for i in range(len(dialogues)):
                    o_file.write('%s^%s\n' % (targets[i], dialogues[i]))
            o_file.close()
    elif(source == 'zhuanzhuan'):
        pass ## TODO
