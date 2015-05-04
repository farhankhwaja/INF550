__author__ = 'FarhanKhwaja'

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import FeatureEngineering

def dayhour(timestr):
    d = datetime.strptime(str(timestr), "%y%m%d%H")
    return [float(d.weekday()), float(d.hour)]


clf = SGDClassifier(loss="log",verbose=2,shuffle=True,alpha=0.000001,n_iter=20)

fe = FeatureEngineering.FeatureEngineering()
fh = FeatureHasher(n_features = 2**20, input_type="string")

train = pd.read_csv("InputFiles/train.csv", chunksize = 1000, iterator = True)
all_classes = np.array([0, 1])

cnt = 1
for chunk in train:
    y_train = chunk["click"]
    chunk = fe.sgdfeature(chunk)
    cv = StratifiedKFold(y_train,shuffle=True,n_folds=4)
    mean_tpr = 0.0

    print('Data Read')

    for i,(train, test) in enumerate(cv):
        print(i,' StratifiedKFold CV ')
        probas_ = clf.partial_fit(chunk[train], y_train[train], classes=all_classes).predict_proba(chunk[test])
        #Compute ROC Curve
        fpr, tpr, threshold = roc_curve(y_train[test],probas_[:,1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    del chunk
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for each data chunk (~10M)')
    plt.legend(loc="lower right")
    name = 'ROCChart_'+str(cnt)+'.png'
    cnt += 1
    plt.savefig(name)
    plt.show()