# utils used in this project
#
# Created by yuanpingzhou at 9/26/18

import time
from contextlib2 import contextmanager

# time cost monitor with seconds
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

# convert probability to label in case of binary classification
def proba2label(data):
    return [1 if(v > 0.5) else 0 for v in data]
