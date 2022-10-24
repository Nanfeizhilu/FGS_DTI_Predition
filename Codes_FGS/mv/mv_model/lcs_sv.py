import numpy as np
import copy

from sklearn.neighbors import NearestNeighbors
from base.mvinbase import MultiViewInductiveModelBase
from base.gipbase import GIPSimBase

from nn.wknkn import WKNKN
from mv.com_sim.combine_sims import Combine_Sims_Ave
from mv.com_sim.combine_sims2 import Combine_Sims_LimbPerDT_2



class LCS_SV(MultiViewInductiveModelBase):
    """
    1. linearly combine the multiple similarities to one similairty matrix
    2. train a DTI predicton mehtod using the combined/fused durg and target similairties
    """
    
    def __init__(self, cs_model=Combine_Sims_Ave(), sv_model=WKNKN(k=7, T=0.8)):
        self.cs_model = cs_model # method to combine multiple similarities to one similarity matrix
        self.sv_model = sv_model # preidiction method using one similarity matrix (single view)
        
        self.copyable_attrs = ['cs_model','sv_model']
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMats, targetMats, cvs=2): 
        self._check_fit_data(intMat, drugMats, targetMats, cvs)
        # get weihgts of each input similarities and combined/fused similarities
        D, self.wd = self.cs_model.combine(drugMats, intMat)
        T, self.wt = self.cs_model.combine(targetMats, intMat.T)
        # train DTI prediciton model based on combined similairties
        self.sv_model.fit(intMat, D, T, cvs)
    #----------------------------------------------------------------------------------------
    
    def predict(self, drugMatTes, targetMatTes):
        self._check_test_data(drugMatTes, targetMatTes)
        # get the combined/fused test similarities
        D_te = np.average(drugMatTes,axis=0,weights=self.wd)
        T_te = np.average(targetMatTes,axis=0,weights=self.wt)
        # predict socres based on combined/fused test similarities
        scores = self.sv_model.predict(D_te, T_te)        
        return scores
    #---------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------- 


class LCS_SV_GIP(LCS_SV, GIPSimBase):
    """
    Different with LCS_SV: add GIP similarity
    """
    
    def fit(self, intMat, drugMats, targetMats, cvs=2): 
        self._check_fit_data(intMat, drugMats, targetMats, cvs)
        
        self._Y = intMat
        
        # add GIP for training set
        self._GIP_d = self.compute_GIP(intMat)
        Sds = self._add_GIP(drugMats, self._GIP_d)
        self._n_dsims = self._n_dsims+1
        self._GIP_t = self.compute_GIP(intMat.T)
        Sts = self._add_GIP(targetMats, self._GIP_t)
        self._n_tsims = self._n_tsims+1
        
        # get weihgts of each input similarities and combined/fused similarities
        D, self.wd = self.cs_model.combine(Sds, intMat)
        T, self.wt = self.cs_model.combine(Sts, intMat.T)
        # train DTI prediciton model based on combined similairties
        self.sv_model.fit(intMat, D, T, cvs)
    #----------------------------------------------------------------------------------------
    
    def predict(self, drugMatTes, targetMatTes):
        GIP_d_te, GIP_t_te = self._compute_GIP_te(drugMatTes[0],targetMatTes[0],self._Y)
        Sds_te = self._add_GIP(drugMatTes, GIP_d_te)
        Sts_te = self._add_GIP(targetMatTes, GIP_t_te)
        
        self._check_test_data(Sds_te, Sts_te)
        
        # get the combined/fused test similarities
        D_te = np.average(Sds_te,axis=0,weights=self.wd)
        T_te = np.average(Sts_te,axis=0,weights=self.wt)
        # predict socres based on combined/fused test similarities
        scores = self.sv_model.predict(D_te, T_te)        
        # print(self.wd,'\t',self.wt)
        return scores
    #---------------------------------------------------------------------------------------- 
    
    def _add_GIP(self, Sds, GIP):
        GIP1 = GIP.reshape(1,GIP.shape[0],GIP.shape[1])
        Sds1 = np.concatenate((Sds,GIP1),axis=0)
        return Sds1
    #---------------------------------------------------------------------------------------- 
    

    def _compute_GIP_te(self, Sd_te, St_te, train_Y):    
        self._n_drugs_te = Sd_te.shape[0]
        self._n_targets_te = St_te.shape[0]
        if self._cvs == 2:
            test_Yd = np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
            test_Yd_impute = self._impute_Y_test(train_Y, test_Yd, Sd_te)
            GIP_d_te = self.compute_GIP_test(train_Y, test_Yd_impute)
            GIP_t_te = self._GIP_t
        elif self._cvs == 3:
            GIP_d_te = self._GIP_d
            test_Yt =np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
            test_Yt_impute = self._impute_Y_test(train_Y.T, test_Yt.T, St_te)
            GIP_t_te = self.compute_GIP_test(train_Y.T, test_Yt_impute)
        elif self._cvs == 4:
            test_Yd = np.zeros((self._n_drugs_te,self._n_targets))
            test_Yd_impute = self._impute_Y_test(train_Y, test_Yd, Sd_te)
            GIP_d_te = self.compute_GIP_test(train_Y, test_Yd_impute)
            test_Yt = np.zeros((self._n_drugs,self._n_targets_te))
            test_Yt_impute = self._impute_Y_test(train_Y.T, test_Yt.T, St_te)
            GIP_t_te = self.compute_GIP_test(train_Y.T, test_Yt_impute)
        return GIP_d_te, GIP_t_te
    #----------------------------------------------------------------------------------------        
    
    
    def _impute_Y_test(self, train_Y, test_Y, S_test, k=5):
        Y = np.zeros(test_Y.shape)
        # S = S_test - np.diag(np.diag(S_test))
        S = S_test
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        neigh.fit(np.zeros((train_Y.shape[0],train_Y.shape[0])))
        knns = neigh.kneighbors(1 - S, return_distance=False)
        
        for d in range(Y.shape[0]):
            ii = knns[d]
            sd = S[d,ii]
            Y[d,:] = sd@train_Y[ii,:] # decay coefficient = 1
            z = np.sum(sd)
            if z>0:
                Y[d,:]/=z                
        return Y
    #----------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------- 

    
class LCS_SV2(LCS_SV):
    """
    Difference with LCS_SV
    The weight Wd and Wt are 2d array; the weight of each test drug/target are determined by kNNs
    """
    
    def __init__(self, cs_model=Combine_Sims_LimbPerDT_2(k=5, rho=0.6), sv_model=WKNKN(k=7, T=0.8)):
        super().__init__(cs_model=cs_model, sv_model=sv_model)
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMats, targetMats, cvs=2):
        self._check_fit_data(intMat, drugMats, targetMats, cvs)
        
        # get weihgts of each input similarities and combined/fused similarities
        self.cs_model_d = copy.deepcopy(self.cs_model)
        self.cs_model_t = copy.deepcopy(self.cs_model)
        self._D, self.Wd = self.cs_model_d.combine(drugMats, intMat)
        self._T, self.Wt = self.cs_model_t.combine(targetMats, intMat.T)
        
        # train DTI prediciton model based on combined similairties
        self.sv_model.fit(intMat, self._D, self._T, cvs)
    #----------------------------------------------------------------------------------------
    
    def predict(self, drugMatTes, targetMatTes):
        self._check_test_data(drugMatTes, targetMatTes)
        # get the combined/fused test similarities
        if self._cvs == 2:
            D_te,_ = self.cs_model_d.combine_test(drugMatTes)
            T_te = self._T
        elif self._cvs == 3:
            D_te = self._D
            T_te,_ = self.cs_model_t.combine_test(targetMatTes)
        elif self._cvs == 4:
            D_te,_ = self.cs_model_d.combine_test(drugMatTes)
            T_te,_ = self.cs_model_t.combine_test(targetMatTes)
        # predict socres based on combined/fused test similarities
        scores = self.sv_model.predict(D_te, T_te)        
        return scores
    #----------------------------------------------------------------------------------------  
   
    
class LCS_SV2_GIP(LCS_SV2, GIPSimBase):
    """
    Difference with LCS_SV2: add GIP
    """
    
    def __init__(self, cs_model=Combine_Sims_LimbPerDT_2(k=5, rho=0.6), sv_model=WKNKN(k=7, T=0.8), beta_d =1.0, beta_t=1.0, theta=0.7):
        super().__init__(cs_model=cs_model, sv_model=sv_model)
        self.beta_d = beta_d # weight coefficient for GIP_d
        self.beta_t = beta_t # weight coefficient for GIP_t
        self.theta = theta # thereshold for beta_d and beta_t, beta_d (beta=t) = 0 if it's value lower than theta
        self.copyable_attrs.extend(['beta_d','beta_t','theta'])
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMats, targetMats, cvs=2):
        self._check_fit_data(intMat, drugMats, targetMats, cvs)
        
        self._Y = intMat
        
        # add GIP for training set
        self._GIP_d = self.compute_GIP(intMat)
        Sds = self._add_GIP(drugMats, self._GIP_d)
        self._n_dsims = self._n_dsims+1
        self._GIP_t = self.compute_GIP(intMat.T)
        Sts = self._add_GIP(targetMats, self._GIP_t)
        self._n_tsims = self._n_tsims+1
        
        # get weihgts of each input similarities and combined/fused similarities
        self.cs_model_d = copy.deepcopy(self.cs_model)
        self.cs_model_t = copy.deepcopy(self.cs_model)
        
        beta_d, beta_t = self.beta_d, self.beta_t
        if self.beta_d < self.theta:
            beta_d = 0
        if self.beta_t < self.theta:
            beta_t = 0
        
        self._D, self.Wd = self.cs_model_d.combine(Sds, intMat, beta_d) # **2
        self._T, self.Wt = self.cs_model_t.combine(Sts, intMat.T, beta_t) # **2
        
        # train DTI prediciton model based on combined similairties
        self.sv_model.fit(intMat, self._D, self._T, cvs)
    #----------------------------------------------------------------------------------------
    
    def predict(self, drugMatTes, targetMatTes):
        # add GIP for test set
        GIP_d_te, GIP_t_te = self._compute_GIP_te(drugMatTes[0],targetMatTes[0],self._Y)
        Sds_te = self._add_GIP(drugMatTes, GIP_d_te)
        Sts_te = self._add_GIP(targetMatTes, GIP_t_te)
        
        self._check_test_data(Sds_te, Sts_te)
        
        
        # get the combined/fused test similarities
        if self._cvs == 2:
            D_te,_ = self.cs_model_d.combine_test(Sds_te)
            T_te = self._T
        elif self._cvs == 3:
            D_te = self._D
            T_te,_ = self.cs_model_t.combine_test(Sts_te)
        elif self._cvs == 4:
            D_te,_ = self.cs_model_d.combine_test(Sds_te)
            T_te,_ = self.cs_model_t.combine_test(Sts_te)
        # predict socres based on combined/fused test similarities
        scores = self.sv_model.predict(D_te, T_te)        
        return scores
    #----------------------------------------------------------------------------------------
    
    def _add_GIP(self, Sds, GIP):
        GIP1 = GIP.reshape(1,GIP.shape[0],GIP.shape[1])
        Sds1 = np.concatenate((Sds,GIP1),axis=0)
        return Sds1
    #---------------------------------------------------------------------------------------- 
    
    def _compute_GIP_te(self, Sd_te, St_te, train_Y):    
        self._n_drugs_te = Sd_te.shape[0]
        self._n_targets_te = St_te.shape[0]
        if self._cvs == 2:
            test_Yd = np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
            test_Yd_impute = self._impute_Y_test(train_Y, test_Yd, Sd_te)
            GIP_d_te = self.compute_GIP_test(train_Y, test_Yd_impute)
            GIP_t_te = self._GIP_t
        elif self._cvs == 3:
            GIP_d_te = self._GIP_d
            test_Yt =np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
            test_Yt_impute = self._impute_Y_test(train_Y.T, test_Yt.T, St_te)
            GIP_t_te = self.compute_GIP_test(train_Y.T, test_Yt_impute)
        elif self._cvs == 4:
            test_Yd = np.zeros((self._n_drugs_te,self._n_targets))
            test_Yd_impute = self._impute_Y_test(train_Y, test_Yd, Sd_te)
            GIP_d_te = self.compute_GIP_test(train_Y, test_Yd_impute)
            test_Yt = np.zeros((self._n_drugs,self._n_targets_te))
            test_Yt_impute = self._impute_Y_test(train_Y.T, test_Yt.T, St_te)
            GIP_t_te = self.compute_GIP_test(train_Y.T, test_Yt_impute)
        return GIP_d_te, GIP_t_te
    #----------------------------------------------------------------------------------------        
    
    
    def _impute_Y_test(self, train_Y, test_Y, S_test, k=5):
        Y = np.zeros(test_Y.shape)
        # S = S_test - np.diag(np.diag(S_test))
        S = S_test
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        neigh.fit(np.zeros((train_Y.shape[0],train_Y.shape[0])))
        knns = neigh.kneighbors(1 - S, return_distance=False)
        
        for d in range(Y.shape[0]):
            ii = knns[d]
            sd = S[d,ii]
            Y[d,:] = sd@train_Y[ii,:] # decay coefficient = 1
            z = np.sum(sd)
            if z>0:
                Y[d,:]/=z                
        return Y
    #----------------------------------------------------------------------------------------
    
    
    



