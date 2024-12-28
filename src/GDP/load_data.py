import numpy as np
import gdp_model as dm
from tensorflow.python.framework import random_seed
import collections

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class DataSet(object):

    """Format the traing data
    """

    def __init__(self,
                 features,
                 dates,
                 censors,
                 #at_risks,
                 feature_groups,
                 seed=None):
        #TODO: check if seed work as expected
        seed1,seed2=random_seed.get_seed(seed)
        np.random.seed(seed1 if seed is None else seed2)

        assert features.shape[0] == dates.shape[0],(
         'features.shape: %s dates.shape: %s' % (features.shape, dates.shape))
        self._patients_num=features.shape[0]
        self._features=features
        self._feature_groups=feature_groups
        self._dates=dates
        self._censors=censors
        #self._at_risks=at_risks
        self._epochs_completed=0
        self._index_in_epoch=0

    @property
    def features(self):
        return self._features
    @property
    def feature_size(self):
        return self._features.shape[1]

    @property
    def feature_groups(self):
        return self._feature_groups

    @property
    def dates(self):
        return self._dates

    @property
    def censors(self):
        return self._censors
#    @property
#    def at_risks(self):
#        return self._at_risks
    @property
    def patients_num(self):
        return self._patients_num

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self,batch_size,shuffle=True):
        start=self._index_in_epoch
        # shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            index_perm=np.arange(self._patients_num)
            np.random.shuffle(index_perm)
            self._features=self._features[index_perm]
            self._dates=self._dates[index_perm]
            self._censors=self._censors[index_perm]
            #self._at_risks=self._at_risks[index_perm]
            #self._at_risks=[self._at_risks[k] for k in index_perm]
            #self._features,self._dates,self._censors,self._at_risks=dm.calc_at_risk(self._features[index_perm],self._dates[index_perm],self._censors[index_perm])

        # Go to the next epoch
        if start + batch_size > self._patients_num:
            # epoch completed
            self._epochs_completed+=1
            # get the remained ones
            rest_patients_num=self._patients_num - start
            features_rest_part=self._features[start:self._patients_num]
            dates_rest_part=self._dates[start:self._patients_num]
            censors_rest_part=self._censors[start:self._patients_num]
            #at_risks_rest_part=self._at_risks[start:self._patients_num]
            #shuffle the data
            if shuffle:
                index_perm1=np.arange(self._patients_num)
                np.random.shuffle(index_perm1)
                self._features=self._features[index_perm1]
                self._dates=self._dates[index_perm1]
                self._censors=self._censors[index_perm1]
                #self._at_risks=self._at_risks[index_perm1] # _at_risks is a python list
                #self._at_risks=[self._at_risks[k] for k in index_perm1]
            # start next epoch
            start = 0
            self._index_in_epoch=batch_size-rest_patients_num
            end=self._index_in_epoch
            features_new_part=self._features[start:end]
            dates_new_part=self._dates[start:end]
            censors_new_part=self._censors[start:end]
            #at_risks_new_part=self._at_risks[start:end]
            features_=np.concatenate((features_rest_part,features_new_part),axis=0)
            dates_=np.concatenate((dates_rest_part,dates_new_part),axis=0)
            censors_=np.concatenate((censors_rest_part,censors_new_part),axis=0)
            return dm.calc_at_risk(features_,dates_,censors_)
            #return np.concatenate((features_rest_part,features_new_part),axis=0),np.concatenate((dates_rest_part,dates_new_part),axis=0),np.concatenate((censors_rest_part,censors_new_part),axis=0),np.concatenate((at_risks_rest_part,at_risks_new_part),axis=0)
        else:
            self._index_in_epoch+=batch_size
            end=self._index_in_epoch
            #return self._features[start:end],self._dates[start:end],self._censors[start:end],self._at_risks[start:end]
            return dm.calc_at_risk(self._features[start:end],self._dates[start:end],self._censors[start:end])


