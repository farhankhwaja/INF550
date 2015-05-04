import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import SGDClassifier
import FeatureEngineering


if __name__ == '__main__':

    start = datetime.now()
    print('\n')

    fe = FeatureEngineering.FeatureEngineering()

    # Train classifier
    clf = SGDClassifier(loss="log",verbose=2,shuffle=True,alpha=0.000001,n_iter=20)
    train = pd.read_csv("InputFiles/train.csv", chunksize = 5000000, iterator = True)
    all_classes = np.array([0, 1])

    for i, chunk in enumerate(train):
        strt_chunk = datetime.now()
        print('Chunk Starting Time : ',strt_chunk)
        print('Reading Chunk %d' %i)

        y_train = chunk['click'].values
        y_train = np.asarray(y_train).ravel()

        chunk = fe.sgdfeature(chunk)

        clf.partial_fit(chunk, y_train, classes=all_classes)
        print('Elapsed Time : ',str(datetime.now() - strt_chunk))
        print('\n----------------------------\n')

    del train

    # Create a submission file
    X_test = pd.read_csv("InputFiles/test.csv")
    X_test = fe.sgdfeature(X_test)
    y_pred = clf.predict_proba(X_test)

    with open("new_featVec_submission.csv", "w") as f:
        f.write("id,click\n")
        for idx, xid in enumerate(y_pred):
            f.write(str(idx+1) + "," + str(xid[1]) + "\n")
    f.close()