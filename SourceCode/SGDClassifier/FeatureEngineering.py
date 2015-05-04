__author__ = 'FarhanKhwaja'

import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.pipeline import Pipeline
from datetime import datetime, date, time


class FeatureEngineering:

    def __int__(self):
        self.y_train = []
        self.X_train = []

    # add two columns for hour and weekday
    def dayhour(timestr):
        d = datetime.strptime(str(timestr), "%y%m%d%H")
        return [float(d.weekday()), float(d.hour)]

    def vwfeature(self,data):

        data['click']=data['click'].values +data['click'].values-1

        ##for Vowpal Wabbit
        data['app'] = data['app_id'].values+data['app_domain'].values+data['app_category'].values
        data['site'] = data['site_id'].values+data['site_domain'].values+data['site_category'].values
        data['device'] = data['device_id'].values+data['device_ip'].values+data['device_model'].values+(data['device_type'].values.astype(str))+(data['device_conn_type'].values.astype(str))
        data['type'] = data['device_type'].values +data['device_conn_type'].values
        data['iden'] = data['app_id'].values +data['site_id'].values +data['device_id'].values
        data['domain'] = data['app_domain'].values +data['site_domain'].values
        data['category'] = data['app_category'].values+data['site_category'].values
        data['pS1'] = data['C1'].values.astype(str)+data['app_id']
        data['pS2'] = data['C14'].values+data['C15'].values+data['C16'].values+data['C17'].values
        data['pS3'] = data['C18'].values+data['C19'].values+data['C20'].values+data['C21'].values
        data['sum'] = data['C1'].values+data['C14'].values+data['C15'].values+data['C16'].values+data['C17'].values\
        +data['C18'].values+data['C19'].values+data['C20'].values+data['C21'].values
        data['pos'] =  data['banner_pos'].values.astype(str)+data['app_category'].values+data['site_category'].values
        data['pS4'] = data['C1'].values-data['C20'].values
        data['ps5'] = data['C14'].values-data['C21'].values

        data['hour'] = data['hour'].map(lambda x: datetime.strptime(x.astype(str),"%y%m%d%H"))
        data['dayoftheweek'] = data['hour'].map(lambda x:  x.weekday())
        data['day'] = data['hour'].map(lambda x:  x.day)
        data['hour'] = data['hour'].map(lambda x:  x.hour)
        day = data['day'].values[len(data)-1]

        return data, np.array(data.columns.values).astype(str)


    def sgdfeature(self,data):

        newdata = pd.DataFrame()

        preproc = Pipeline([('fh',FeatureHasher( n_features=2**20,input_type='string'))])

        ##for SGDClassifier
        newdata['app_id_specs'] = data['app_id'].values+data['app_domain'].values+data['app_category'].values
        newdata['app_dom_specs'] = data['app_domain'].values+data['app_category'].values
        newdata['site_id_specs'] = data['site_id'].values+data['site_domain'].values+data['site_category'].values
        newdata['site_dom_specs'] = data['site_domain'].values+data['site_category'].values
        newdata['type'] = data['device_type'].values +data['device_conn_type'].values
        newdata['domain'] = data['app_domain'].values +data['site_domain'].values
        newdata['category'] = data['app_category'].values+data['site_category'].values
        newdata['pos_cat'] =  data['banner_pos'].values.astype(str)+data['app_category'].values+data['site_category'].values
        newdata['pos_dom'] =  data['banner_pos'].values.astype(str)+data['app_domain'].values+data['site_domain'].values
        newdata['timestamp'] = data['hour'].map(lambda x: datetime.strptime(x.astype(str),"%y%m%d%H"))
        newdata['dayoftheweek'] = newdata['timestamp'].map(lambda x:  x.weekday)
        newdata['day'] = newdata['timestamp'].map(lambda x:  x.day)
        newdata['hour'] = newdata['timestamp'].map(lambda x:  x.hour)
        newdata = newdata.drop('timestamp',axis=1)
        newdata = newdata.astype(str)
        del data
        X_dict = np.asarray(newdata)

        self.X_train = preproc.fit_transform(X_dict)

        return self.X_train


