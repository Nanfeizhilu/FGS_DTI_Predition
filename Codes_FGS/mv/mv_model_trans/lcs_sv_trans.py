import numpy as np
import copy

from sklearn.neighbors import NearestNeighbors
from base.mvtransbase import MultiViewTransductiveModelBase
from base.gipbase import GIPSimBase

from model_trans.mf.nrlmf_trans import NRLMF_TRANS
from mv.com_sim.combine_sims import Combine_Sims_Ave
from mv.com_sim.combine_sims2 import Combine_Sims_LimbPerDT_2
from base.splitdata import split_train_test_set_mv

from sklearn.metrics.pairwise import rbf_kernel

class LCS_SV_TRANS(MultiViewTransductiveModelBase):
    """
    1. linearly combine the multiple similarities to one similairty matrix
    2. train a DTI predicton mehtod using the combined/fused durg and target similairties
    """
    
    def __init__(self, cs_model=Combine_Sims_Ave(), sv_model=NRLMF_TRANS(cfix=5, K1=5, K2=5, num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0)):
        self.cs_model = cs_model # method to combine multiple similarities to one similarity matrix
        self.sv_model = sv_model # preidiction method using one similarity matrix (single view)
        
        self.copyable_attrs = ['cs_model','sv_model']
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMats, targetMats, test_indices, cvs=2):
        self._check_fit_data(intMat, drugMats, targetMats, test_indices, cvs)
        # get weihgts of each input similarities and combined/fused similarities
        D, self.wd = self.cs_model.combine(drugMats, intMat)
        T, self.wt = self.cs_model.combine(targetMats, intMat.T)
        # train DTI prediciton model based on combined similairties
        S_te = self.sv_model.fit(intMat, D, T, test_indices, cvs)
        return S_te
    #----------------------------------------------------------------------------------------
    
    def _get_prediction_trainset(self):
        S = self.sv_model._get_prediction_trainset()
        return S
    #----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------   
    
    
class LCS_SV_GIP_TRANS(LCS_SV_TRANS, GIPSimBase):
    """
    Different with LCS_SV: add GIP similarity
    """
    def fit(self, intMat, drugMats, targetMats, test_indices, cvs=2):
        self._check_fit_data(intMat, drugMats, targetMats, test_indices, cvs)
        
        if self._cvs == 1:
            Yd = intMat
            Yt = intMat.T
        elif self._cvs == 2:
            test_d = test_indices # test drug indices
            all_d = np.arange(self._n_drugs)
            train_d = np.setdiff1d(all_d, test_d)
            Yd = self._impute_Y_test(intMat, drugMats[0], train_d, test_d)
            Yt = intMat.T  
        elif self._cvs == 3:
            Yd = intMat
            test_t = test_indices
            all_t = np.arange(self._n_targets)
            train_t = np.setdiff1d(all_t, test_t)
            Yt = self._impute_Y_test(intMat.T, targetMats[0], train_t, test_t)
        elif self._cvs == 4: 
            test_d,test_t = test_indices
            all_d = np.arange(self._n_drugs)
            train_d = np.setdiff1d(all_d, test_d)
            Yd = self._impute_Y_test(intMat, drugMats[0], train_d, test_d)
            all_t = np.arange(self._n_targets)
            train_t = np.setdiff1d(all_t, test_t)
            Yt = self._impute_Y_test(intMat.T, targetMats[0], train_t, test_t)
        
        GIP_d = self.compute_GIP(Yd)
        Sds = self._add_GIP(drugMats, GIP_d)
        self._n_dsims = self._n_dsims+1
        GIP_t = self.compute_GIP(Yt)
        Sts = self._add_GIP(targetMats, GIP_t)
        self._n_tsims = self._n_tsims+1      
        
        # get weihgts of each input similarities and combined/fused similarities
        D, self.wd = self.cs_model.combine(Sds, intMat)
        T, self.wt = self.cs_model.combine(Sts, intMat.T)
        # train DTI prediciton model based on combined similairties
        S_te = self.sv_model.fit(intMat, D, T, test_indices, cvs)
        return S_te
    #----------------------------------------------------------------------------------------
    
    def _add_GIP(self, Sds, GIP):
        GIP1 = GIP.reshape(1,GIP.shape[0],GIP.shape[1])
        Sds1 = np.concatenate((Sds,GIP1),axis=0)
        return Sds1
    #---------------------------------------------------------------------------------------- 
     
    def _impute_Y_test(self, intMat, Sd, train_d, test_d, k=5):
        Y = np.copy(intMat)
        S = Sd - np.diag(np.diag(Sd))
        S = S[:,train_d] # find kNNs from training drugs
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        neigh.fit(np.zeros((S.shape[1],S.shape[1])))
        knns = neigh.kneighbors(1 - S, return_distance=False)
        
        for d in test_d:
            ii = knns[d]
            sd = S[d,ii]
            Y[d,:] = sd@intMat[ii,:]
            z = np.sum(sd)
            if z>0:
                Y[d,:]/=z                
        return Y
    #----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------


class LCS_SV2_TRANS(LCS_SV_TRANS):
    """
    Difference with LCS_SV_TRANS
    The weight Wd and Wt are 2d array; the weight of each test drug/target are determined by kNNs
    """
    
    def __init__(self, cs_model=Combine_Sims_LimbPerDT_2(k=5, rho=0.6), sv_model=NRLMF_TRANS(cfix=5, K1=5, K2=5, num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0)):
        super().__init__(cs_model=cs_model, sv_model=sv_model)
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMats, targetMats, test_indices, cvs=2):
        self._check_fit_data(intMat, drugMats, targetMats, test_indices, cvs)
        # get weihgts of each input similarities and combined/fused similarities
        self.cs_model_d = copy.deepcopy(self.cs_model)
        self.cs_model_t = copy.deepcopy(self.cs_model)
        
        D = np.zeros(drugMats[0].shape)
        T = np.zeros(targetMats[0].shape)
        train_dMats, train_tMats, train_Y, test_dMats, test_tMats, train_d, test_d, train_t, test_t = self._split_train_test_data(intMat, drugMats, targetMats, test_indices)
        # combine similarity for training data
        D_train, self.Wd = self.cs_model_d.combine(train_dMats, train_Y)
        T_train, self.Wt = self.cs_model_t.combine(train_tMats, train_Y.T)
        
        # combine similarity for test data and construct the all fused similarity matrix
        if self._cvs == 1:
            D = D_train
            T = T_train
        elif self._cvs == 2:
            D[np.ix_(train_d,train_d)] = D_train
            D_test,_ = self.cs_model_d.combine_test(test_dMats)
            D[np.ix_(test_d,train_d)] = D_test
            D[np.ix_(train_d,test_d)] = D_test.T
            T = T_train
        elif self._cvs == 3:
            D = D_train
            T[np.ix_(train_t,train_t)] = T_train
            T_test,_ = self.cs_model_t.combine_test(test_tMats)
            T[np.ix_(test_t,train_t)] = T_test
            T[np.ix_(train_t,test_t)] = T_test.T
        elif self._cvs == 4:
            D[np.ix_(train_d,train_d)] = D_train
            D_test,_ = self.cs_model_d.combine_test(test_dMats)
            D[np.ix_(test_d,train_d)] = D_test
            D[np.ix_(train_d,test_d)] = D_test.T
            T[np.ix_(train_t,train_t)] = T_train
            T_test,_ = self.cs_model_t.combine_test(test_tMats)
            T[np.ix_(test_t,train_t)] = T_test
            T[np.ix_(train_t,test_t)] = T_test.T
        S_te = self.sv_model.fit(intMat, D, T, test_indices, cvs)
        return S_te
    #----------------------------------------------------------------------------------------

    def _split_train_test_data(self, intMat, drugMats, targetMats, test_indices):
        if self._cvs == 1: 
            train_dMats = drugMats
            train_tMats = targetMats 
            train_Y = intMat
            train_d, test_d, train_t, test_t = None, None, None, None
            test_dMats, test_tMats = None, None
            # print (self.__class__.__name__, " can not handle S1 prediciton setting!!")
            # return None
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
        if self._cvs in [2,3,4]:
            train_dMats, train_tMats, train_Y, test_dMats, test_tMats, test_Y = split_train_test_set_mv(train_d,train_t,test_d,test_t, intMat, drugMats, targetMats, self._cvs)
        return train_dMats, train_tMats, train_Y, test_dMats, test_tMats, train_d, test_d, train_t, test_t
    #----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------


class LCS_SV2_GIP_TRANS(LCS_SV2_TRANS, GIPSimBase):
    """
    Difference with LCS_SV2_TRANS: add GIP
    """    
    def __init__(self, cs_model=Combine_Sims_LimbPerDT_2(k=5, rho=0.6), sv_model=NRLMF_TRANS(cfix=5, K1=5, K2=5, num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0), beta_d =1.0, beta_t=1.0, theta=0.7):
        super().__init__(cs_model=cs_model, sv_model=sv_model)
        self.beta_d = beta_d # weight coefficient for GIP_d
        self.beta_t = beta_t # weight coefficient for GIP_t
        self.theta = theta # thereshold for beta_d and beta_t, beta_d (beta=t) = 0 if it's value lower than theta
        self.copyable_attrs.extend(['beta_d','beta_t','theta'])
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMats, targetMats, test_indices, cvs=2):
        self._check_fit_data(intMat, drugMats, targetMats, test_indices, cvs)
        # get weihgts of each input similarities and combined/fused similarities
        self.cs_model_d = copy.deepcopy(self.cs_model)
        self.cs_model_t = copy.deepcopy(self.cs_model)
        
        D = np.zeros(drugMats[0].shape)
        T = np.zeros(targetMats[0].shape)
        
        # add GIP
        if self._cvs == 1:
            Yd = intMat
            Yt = intMat.T
        elif self._cvs == 2:
            test_d = test_indices # test drug indices
            all_d = np.arange(self._n_drugs)
            train_d = np.setdiff1d(all_d, test_d)
            Yd = self._impute_Y_test(intMat, drugMats[0], train_d, test_d)
            Yt = intMat.T  
        elif self._cvs == 3:
            Yd = intMat
            test_t = test_indices
            all_t = np.arange(self._n_targets)
            train_t = np.setdiff1d(all_t, test_t)
            Yt = self._impute_Y_test(intMat.T, targetMats[0], train_t, test_t)
        elif self._cvs == 4: 
            test_d,test_t = test_indices
            all_d = np.arange(self._n_drugs)
            train_d = np.setdiff1d(all_d, test_d)
            Yd = self._impute_Y_test(intMat, drugMats[0], train_d, test_d)
            all_t = np.arange(self._n_targets)
            train_t = np.setdiff1d(all_t, test_t)
            Yt = self._impute_Y_test(intMat.T, targetMats[0], train_t, test_t)
        GIP_d = self.compute_GIP(Yd)
        Sds = self._add_GIP(drugMats, GIP_d)
        self._n_dsims = self._n_dsims+1
        GIP_t = self.compute_GIP(Yt)
        Sts = self._add_GIP(targetMats, GIP_t)
        self._n_tsims = self._n_tsims+1      

        train_dMats, train_tMats, train_Y, test_dMats, test_tMats, train_d, test_d, train_t, test_t, x_dMats, x_tMats = self._split_train_test_data(intMat, Sds, Sts, test_indices)


        beta_d, beta_t = self.beta_d, self.beta_t
        if self.beta_d < self.theta:
            beta_d = 0
        if self.beta_t < self.theta:
            beta_t = 0
        
        # combine similarity for training data
        D_train, self.Wd = self.cs_model_d.combine(train_dMats, train_Y, beta_d)
        T_train, self.Wt = self.cs_model_t.combine(train_tMats, train_Y.T, beta_t)
        
        # combine similarity for test data and construct the all fused similarity matrix
        if self._cvs == 1:
            D = D_train
            T = T_train
        elif self._cvs == 2:
            D[np.ix_(train_d,train_d)] = D_train
            D_test, Wd_te = self.cs_model_d.combine_test(test_dMats)
            D[np.ix_(test_d,train_d)] = D_test
            D[np.ix_(train_d,test_d)] = D_test.T
            T = T_train
            """! setting similarity between test drugs/targets as Dx is worse than setting them as 0"""
            # Dx = self.cs_model_d._combine_sim(Wd_te, x_dMats)   
            # D[np.ix_(test_d,test_d)] = Dx
        elif self._cvs == 3:
            D = D_train
            T[np.ix_(train_t,train_t)] = T_train
            T_test, Wt_te = self.cs_model_t.combine_test(test_tMats)
            T[np.ix_(test_t,train_t)] = T_test
            T[np.ix_(train_t,test_t)] = T_test.T
            
            # Tx = self.cs_model_d._combine_sim(Wt_te, x_tMats)  
            # T[np.ix_(test_t,test_t)] = Tx
        elif self._cvs == 4:
            D[np.ix_(train_d,train_d)] = D_train
            D_test,Wd_te = self.cs_model_d.combine_test(test_dMats)
            D[np.ix_(test_d,train_d)] = D_test
            D[np.ix_(train_d,test_d)] = D_test.T
            T[np.ix_(train_t,train_t)] = T_train
            T_test,Wt_te = self.cs_model_t.combine_test(test_tMats)
            T[np.ix_(test_t,train_t)] = T_test
            T[np.ix_(train_t,test_t)] = T_test.T
            
            # Dx = self.cs_model_d._combine_sim(Wd_te, x_dMats)  
            # D[np.ix_(test_d,test_d)] = Dx
            # Tx = self.cs_model_d._combine_sim(Wt_te, x_tMats)  
            # T[np.ix_(test_t,test_t)] = Tx           
        S_te = self.sv_model.fit(intMat, D, T, test_indices, cvs)
        return S_te
    #----------------------------------------------------------------------------------------

    def _split_train_test_data(self, intMat, drugMats, targetMats, test_indices):
        if self._cvs == 1: 
            train_dMats = drugMats
            train_tMats = targetMats 
            train_Y = intMat
            train_d, test_d, train_t, test_t = None, None, None, None
            test_dMats, test_tMats = None, None
            x_dMats, x_tMats = None, None
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
        if self._cvs in [2,3,4]:
            train_dMats, train_tMats, train_Y, test_dMats, test_tMats, test_Y, x_dMats, x_tMats = self._split_train_test_set_mv(train_d,train_t,test_d,test_t, intMat, drugMats, targetMats)
        return train_dMats, train_tMats, train_Y, test_dMats, test_tMats, train_d, test_d, train_t, test_t, x_dMats, x_tMats
    #----------------------------------------------------------------------------------------

    def _split_train_test_set_mv(self, train_d,train_t,test_d,test_t, yMat, dMats, tMats): 
        if self._cvs==1:
            print ("Inductive Learning cannot handle Setting 1 !!!")
            return None
        n_dsims = dMats.shape[0]
        idx_d0 = np.arange(n_dsims, dtype = int)
        n_tsims = tMats.shape[0]
        idx_t0 = np.arange(n_tsims, dtype = int)
        x_dMats, x_tMats = None, None # similarity between test drugs/targets
        if self._cvs==2:
            train_dMats = dMats[np.ix_(idx_d0,train_d,train_d)]
            test_dMats = dMats[np.ix_(idx_d0,test_d,train_d)]
            x_dMats = dMats[np.ix_(idx_d0,test_d,test_d)]
            train_tMats = tMats
            test_tMats  = tMats
            train_Y = yMat[train_d,:]
            test_Y  = yMat[test_d,:]
        elif self._cvs==3:
            train_dMats=dMats
            test_dMats=dMats
            train_tMats = tMats[np.ix_(idx_t0,train_t,train_t)]
            test_tMats  = tMats[np.ix_(idx_t0,test_t,train_t)] 
            x_tMats  = tMats[np.ix_(idx_t0,test_t,test_t)] 
            train_Y = yMat[:,train_t]
            test_Y  = yMat[:,test_t] 
        elif self._cvs==4:
            train_dMats = dMats[np.ix_(idx_d0,train_d,train_d)]
            test_dMats = dMats[np.ix_(idx_d0,test_d,train_d)]
            x_dMats = dMats[np.ix_(idx_d0,test_d,test_d)]
            train_tMats = tMats[np.ix_(idx_t0,train_t,train_t)]
            test_tMats  = tMats[np.ix_(idx_t0,test_t,train_t)]
            x_tMats  = tMats[np.ix_(idx_t0,test_t,test_t)] 
            train_Y = yMat[np.ix_(train_d,train_t)]
            test_Y  = yMat[np.ix_(test_d,test_t)]
            
        return train_dMats, train_tMats, train_Y, test_dMats, test_tMats, test_Y, x_dMats, x_tMats
    #---------------------------------------------------------------------------------------------------



    def _add_GIP(self, Sds, GIP):
        GIP1 = GIP.reshape(1,GIP.shape[0],GIP.shape[1])
        Sds1 = np.concatenate((Sds,GIP1),axis=0)
        return Sds1
    #---------------------------------------------------------------------------------------- 
     
    def _impute_Y_test(self, intMat, Sd, train_d, test_d, k=5):
        Y = np.copy(intMat)
        S = Sd - np.diag(np.diag(Sd))
        S = S[:,train_d] # find kNNs from training drugs
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        neigh.fit(np.zeros((S.shape[1],S.shape[1])))
        knns = neigh.kneighbors(1 - S, return_distance=False)
        
        for d in test_d:
            ii = knns[d]
            sd = S[d,ii]
            Y[d,:] = sd@intMat[ii,:]
            z = np.sum(sd)
            if z>0:
                Y[d,:]/=z                
        return Y
    #----------------------------------------------------------------------------------------
    
    
    