import numpy as np


from base.transbase import TransductiveModelBase
from sklearn.neighbors import NearestNeighbors # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
from base.splitdata import split_train_test_set
from nn.wknknri4 import WKNKNRI4_


class WKNKNRI4__TRANS(TransductiveModelBase):
    """
    This is the updated WKNKNRI method in Applied Intelligence
    Difference with WKNKNIR: 
        1. recovery interaction by using NCd as weighting matrix
        2. Ydt are used for all settings (S2, S3, S4)
        3. add exponent for similarities with lower local imbalance of corresponding intMat submatrix
    """
    def __init__(self, k=5, kr=5, T=0.7):
        self.k = k
        self.kr = kr # the k for recovery, which is same with kr for simplicity
        self.T = T # η

        self.copyable_attrs = ['k','kr','T']
    #--------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMat, targetMat, test_indices, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, test_indices, cvs)
        train_dMat, train_tMat, train_Y, test_dMat, test_tMat, _ = self._split_train_test_data(intMat, drugMat, targetMat, test_indices)
        
        model = WKNKNRI4_(self.k, self.kr, self.T)
        model.fit(train_Y, train_dMat, train_tMat, self._cvs)
        scores = model.predict(test_dMat, test_tMat)
        return scores
    #--------------------------------------------------------------------------------------
    
    def _split_train_test_data(self, intMat, drugMat, targetMat, test_indices):
        if self._cvs == 1: 
            print (self.__class__.__name__, " can not handle S1 prediciton setting!!")
            return None
        elif self._cvs == 2:
            test_d = test_indices # test drug indices
            all_d = np.arange(self._n_drugs)
            train_d = np.setdiff1d(all_d, test_d)
            test_t = np.arange(self._n_targets)
            train_t = np.arange(self._n_targets)
        elif self._cvs == 3:
            test_d = np.arange(self._n_drugs)
            train_d = np.arange(self._n_drugs)
            test_t = test_indices
            all_t = np.arange(self._n_targets)
            train_t = np.setdiff1d(all_t, test_t)
        elif self._cvs == 4: 
            test_d,test_t = test_indices
            all_d = np.arange(self._n_drugs)
            train_d = np.setdiff1d(all_d, test_d)
            all_t = np.arange(self._n_targets)
            train_t = np.setdiff1d(all_t, test_t)
        train_dMat, train_tMat, train_Y, test_dMat, test_tMat, test_Y = split_train_test_set(train_d,train_t,test_d,test_t, intMat, drugMat, targetMat, self._cvs)
        return train_dMat, train_tMat, train_Y, test_dMat, test_tMat, test_Y
    #--------------------------------------------------------------------------------------    
#--------------------------------------------------------------------------------------    
