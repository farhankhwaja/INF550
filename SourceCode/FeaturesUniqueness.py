__author__ = 'FarhanKhwaja'

from datetime import datetime
import pandas as pd
from sklearn.externals import joblib
import FeatureEngineering
import numpy as np

if __name__ == '__main__':

    start = datetime.now()
    print('Start time : ',start)

train = pd.read_csv("InputFiles/train.csv", chunksize = 5000000, iterator = True)
uniqVal = {}

fe = FeatureEngineering.FeatureEngineering()

for i, chunk in enumerate(train):
    strt_chunk = datetime.now()
    print('Chunk Starting Time : ',strt_chunk)
    print('Reading Chunk %d' %i)
    names = chunk.columns.values
    for key in names:
        if key not in ['id','target'] and i == 0:
            values = chunk[key]
            uniqVal.setdefault(key,set())
            uniqVal[key] = set(x.strip('\n') for x in str(list(pd.Series(values.values.ravel()).unique())).replace('[','').replace(']','').split(','))
        elif key not in ['id','target']:
            values = chunk[key]
            uniqVal.setdefault(key,set())
            uniqVal[key].union(set(x.strip('\n') for x in str(list(pd.Series(values.values.ravel()).unique())).replace('[','').replace(']','').replace(' ','').split(',')))
        print('Elapsed Time : ',str(datetime.now() - strt_chunk))
        print('\n----------------------------\n')

joblib.dump(uniqVal,'uniqVal.pkl')