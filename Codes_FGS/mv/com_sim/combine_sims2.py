import numpy as np
from base.csbase import Combine_Sims_Base
from sklearn.neighbors import NearestNeighbors

""" combine multiple similarities using the weight of each drug/target"""

class Combine_Sims_LimbPerDT(Combine_Sims_Base):
    def __init__(self, k=5):
        self.k = k
        self.copyable_attrs=['k']
    #---------------------------------------------------------------------------------------- 
    
    def combine(self, Ss, Y):
        self._num_sims = Ss.shape[0]
        self._n = Ss.shape[1] # the number of drugs/targets
        self.W = self._compute_weights(Ss, Y) # W: num_sims,n
        S = self._combine_sim(self.W, Ss) 
        return S, self.W
    #---------------------------------------------------------------------------------------- 
    
    def combine_test(self, Ss_te):
        # combine the test similarities, Ss is the test similairties
        self._check_test_sim(Ss_te)
        self._m = Ss_te.shape[1] # the number of test drugs/targets
        neigh = NearestNeighbors(n_neighbors=self.k, metric='precomputed')
        neigh.fit(np.zeros((self._n,self._n)))
        W_te = np.zeros((self._num_sims,self._m), dtype=float) 
        
        for i in range(self._num_sims):
            knn = neigh.kneighbors(1 - Ss_te[i], return_distance=False) # knn.shape = (m,k)
            U = self.W[i,:][knn] # U.shape = (m,k)
            W_te[i,:] = np.mean(U, axis=1)
        
        sum_W_rows = np.sum(W_te, axis=0)
        W_te = W_te/sum_W_rows[None,:] # the sum of each columns in W is 1
        S_te = self._combine_sim(W_te, Ss_te)        
        return S_te, W_te
    #----------------------------------------------------------------------------------------    
    
    def _combine_sim(self, W, Ss):
        # W.shape= num_sims,n, Ss.shape=num_sims,n,n
        W1 = W[:,:,None] # W1.shape = num_sims,n,1
        S = Ss*W1
        S = np.sum(S,axis=0)
        return S
    #---------------------------------------------------------------------------------------- 
    
    def _check_test_sim(self, Ss):
        if Ss.shape[0] != self._num_sims:
            raise RuntimeError("The number of similairties in Ss ({}) is not same with self._num_sims ({})!!".format(Ss.shape[0], self._num_sims))
        if self._n != Ss.shape[2]:
            raise RuntimeError("The self._n:{} is not comparable with Ss's column {}!!".format(self._n, Ss.shape[2]))
    #----------------------------------------------------------------------------------------    
    
    def _compute_weights(self, Ss, Y):
        W = np.zeros((self._num_sims,self._n), dtype=float) 
        for i in range(self._num_sims):
            S1 = Ss[i] - np.diag(np.diag(Ss[i])) # set diagnol elements to zeros
            C = self._cal_limb(S1, Y, self.k)
            W[i,:] = np.sum(C, axis=1)+1.0/self.k 
            # W[i,h] is the difficulty of d_h in i-th Sim, 1.0/self.k is an smoothing parameter ensuring none zero in W

        W = 1/W  # np.exp(-1*W) has similar performance 
        sum_W_rows = np.sum(W, axis=0)
        W = W/sum_W_rows[None,:] # the sum of each columns in W is 1
        return W
    #---------------------------------------------------------------------------------------- 
    
    def _cal_limb(self, S, Y, k):
        """ S is similarity matrix whose dignoal elememets are zeros"""
        
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        neigh.fit(np.zeros(S.shape))
        knns = neigh.kneighbors(1 - S, return_distance=False)
        
        C = np.zeros(Y.shape, dtype=float)
        for i in range(Y.shape[0]):
            ii = knns[i]
            for j in range(Y.shape[1]):
                if Y[i,j] == 1: # only consider "1" 
                    C[i,j] = k-np.sum(Y[ii,j])
        C = C/k
        #milb = np.sum(C)/np.sum(Y)
        
        return C
    #---------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------- 


class Combine_Sims_LimbPerDT_1(Combine_Sims_Base):
    def __init__(self, k=5):
        self.k = k
        self.copyable_attrs=['k']
    #---------------------------------------------------------------------------------------- 
    
    def combine(self, Ss, Y):
        self._num_sims = Ss.shape[0]
        self._n = Ss.shape[1] # the number of drugs/targets
        self.W = self._compute_weights(Ss, Y) # W: num_sims,n
        S = self._combine_sim(self.W, Ss) 
        
    
        S[S>=1] = 1.0
        
        return S, self.W
    #---------------------------------------------------------------------------------------- 
    
    def combine_test(self, Ss_te):
        # combine the test similarities, Ss is the test similairties
        self._check_test_sim(Ss_te)
        self._m = Ss_te.shape[1] # the number of test drugs/targets
        neigh = NearestNeighbors(n_neighbors=self.k, metric='precomputed')
        neigh.fit(np.zeros((self._n,self._n)))
        W_te = np.zeros((self._num_sims,self._m), dtype=float) 
        
        for i in range(self._num_sims):
            knn = neigh.kneighbors(1 - Ss_te[i], return_distance=False) # knn.shape = (m,k)
            U = self.W[i,:][knn] # U.shape = (m,k)
            W_te[i,:] = np.mean(U, axis=1)
        
        sum_W_rows = np.sum(W_te, axis=0)
        if np.any(sum_W_rows==0):
            print(W_te)
        W_te = W_te/sum_W_rows[None,:] # the sum of each columns in W is 1
        S_te = self._combine_sim(W_te, Ss_te)
        
        # S_te_sum = np.sum(S_te, axis=1)
        # if np.any(S_te_sum==0):
        #     print(S_te_sum)
        S_te[S_te>1] = 1
        return S_te, W_te
    #----------------------------------------------------------------------------------------    
    
    def _combine_sim(self, W, Ss):
        # W.shape= num_sims,n, Ss.shape=num_sims,n,n
        W1 = W[:,:,None] # W1.shape = num_sims,n,1
        S = Ss*W1
        S = np.sum(S,axis=0)
        return S
    #---------------------------------------------------------------------------------------- 
    
    def _check_test_sim(self, Ss):
        if Ss.shape[0] != self._num_sims:
            raise RuntimeError("The number of similairties in Ss ({}) is not same with self._num_sims ({})!!".format(Ss.shape[0], self._num_sims))
        if self._n != Ss.shape[2]:
            raise RuntimeError("The self._n:{} is not comparable with Ss's column {}!!".format(self._n, Ss.shape[2]))
    #----------------------------------------------------------------------------------------    
    
    def _compute_weights(self, Ss, Y):
        W = np.zeros((self._num_sims,self._n), dtype=float) 
        wg = np.zeros(self._num_sims, dtype=float)  # the global local imbalance based weight
        for i in range(self._num_sims):
            S1 = Ss[i] - np.diag(np.diag(Ss[i])) # set diagnol elements to zeros
            milb, C = self._cal_limb(S1, Y, self.k)
            wg[i] = 1- milb
            
            idx1 = np.where(Y==1)
            C[idx1] = 1-C[idx1]
            W[i,:] = np.sum(C, axis=1) #+1.0/(self.k*self._num_sims)
            # W[i,h] is the easiness of d_h in i-th Sim, 1.0/self.k is an smoothing parameter ensuring none zero in W
         
        sum_wg = np.sum(wg)
            
        sum_W_rows = np.sum(W, axis=0)  
        idx0 = np.where(sum_W_rows==0)[0] # indices of durgs whose sum of weight is zero
        sum_W_rows[idx0] = sum_wg
        W[:,idx0] = wg[:,None]
        W = W/sum_W_rows[None,:] # the sum of each columns in W is 1
        return W
    #---------------------------------------------------------------------------------------- 
    
    def _cal_limb(self, S, Y, k):
        """ S is similarity matrix whose dignoal elememets are zeros"""
        
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        neigh.fit(np.zeros(S.shape))
        knns = neigh.kneighbors(1 - S, return_distance=False)
        
        C = np.zeros(Y.shape, dtype=float)
        for i in range(Y.shape[0]):
            ii = knns[i]
            for j in range(Y.shape[1]):
                if Y[i,j] == 1: # only consider "1" 
                    C[i,j] = k-np.sum(Y[ii,j])
        C = C/k
        milb = np.sum(C)/np.sum(Y)
        
        return milb, C
    #---------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------- 

class Combine_Sims_LimbPerDT_2(Combine_Sims_LimbPerDT_1):
    """ set some samllest weights in each row (of each drug) to zeros"""
    def __init__(self, k=5, rho = 0.6):
        self.k = k
        self.rho = rho # the percetage of weghts of similarities removed
        self.copyable_attrs=['k', 'rho']
    #---------------------------------------------------------------------------------------- 
    
    def combine_test(self, Ss_te):
        # combine the test similarities, Ss is the test similairties
        self._check_test_sim(Ss_te)
        self._m = Ss_te.shape[1] # the number of test drugs/targets
        neigh = NearestNeighbors(n_neighbors=self.k, metric='precomputed')
        neigh.fit(np.zeros((self._n,self._n)))
        W_te = np.zeros((self._num_sims,self._m), dtype=float) 
        
        for i in range(self._num_sims):
            knn = neigh.kneighbors(1 - Ss_te[i], return_distance=False) # knn.shape = (m,k)
            U = self.W[i,:][knn] # U.shape = (m,k)
            W_te[i,:] = np.mean(U, axis=1)
        
        # sum_W_rows = np.sum(W_te, axis=0)
        
        """ !!! No test weights are all zeros"""
        # idx0 = np.where(sum_W_rows==0)[0] # indices of durgs whose sum of weight is zero
        # sum_W_rows[idx0] = self.sum_wg
        # W_te[:,idx0] = self.wg[:,None]       
        # if len(idx0) > 0:
        #     print("weight", idx0)
        
        # set smaller rn_sims weights in each column of W to zero
        if self.rn_sims>0:
            idx_par = np.argpartition(W_te, kth=self.rn_sims, axis=0)
            W_te[idx_par[:self.rn_sims,:],np.arange(W_te.shape[1])[None,:]] = 0   
            
        sum_W_rows = np.sum(W_te, axis=0) # recompute the sum of each columns
        sum_W_rows[sum_W_rows==0] = 1 # ensure no zero vlaues in sum_W_rows, as it will be used as denominator
        W_te = W_te/sum_W_rows[None,:] # the sum of each columns in W is 1

        
        S_te = self._combine_sim(W_te, Ss_te)  
        
        # check if any row of S_te are all 0s
        # sum_S_rows = np.sum(S_te, axis=1)
        # idx0 = np.where(sum_S_rows==0)[0]
        # if len(idx0) > 0:
        #     print("sim", idx0)
        #     S_te[idx0] = np.average(Ss_te[:,idx0,:],axis=0,weights=self.wg)
        S_te[S_te>1] = 1
        return S_te, W_te
    #----------------------------------------------------------------------------------------    
    
    
    # def _compute_weights(self, Ss, Y):
    #     self.rn_sims = int(self.rho*self._num_sims)
        
    #     W = np.zeros((self._num_sims,self._n), dtype=float) 
    #     self.wg = np.zeros(self._num_sims, dtype=float)  # the global local imbalance based weight
    #     for i in range(self._num_sims):
    #         S1 = Ss[i] - np.diag(np.diag(Ss[i])) # set diagnol elements to zeros
    #         milb, C = self._cal_limb(S1, Y, self.k)
    #         self.wg[i] = 1- milb
            
    #         idx1 = np.where(Y==1)
    #         C[idx1] = 1-C[idx1]
    #         W[i,:] = np.sum(C, axis=1) #+1.0/(self.k*self._num_sims)
    #         # W[i,h] is the easiness of d_h in i-th Sim, 1.0/self.k is an smoothing parameter ensuring none zero in W
        
    #     # set smaller rn_sims weights in each column of W to zero
    #     if self.rn_sims>0:
    #         idx_par = np.argpartition(self.wg, kth=self.rn_sims)
    #         self.wg[idx_par[:self.rn_sims]] = 0
    #     self.sum_wg = np.sum(self.wg)
            
    #     sum_W_rows = np.sum(W, axis=0)  
    #     idx0 = np.where(sum_W_rows==0)[0] # indices of durgs whose sum of weight is zero
    #     sum_W_rows[idx0] = self.sum_wg
    #     W[:,idx0] = self.wg[:,None]
        
    #     # set smaller rn_sims weights in each column of W to zero
    #     if self.rn_sims>0:
    #         idx_par = np.argpartition(W, kth=self.rn_sims, axis=0)
    #         W[idx_par[:self.rn_sims,:],np.arange(W.shape[1])[None,:]] = 0   
    #         sum_W_rows = np.sum(W, axis=0) # recompute the sum of each columns
            
    #     W = W/sum_W_rows[None,:] # the sum of each columns in W is 1
    #     return W
    # #---------------------------------------------------------------------------------------- 
    
    
    def _compute_weights(self, Ss, Y):
        self.rn_sims = int(self.rho*self._num_sims)
        
        W = np.zeros((self._num_sims,self._n), dtype=float) 
        self.wg = np.zeros(self._num_sims, dtype=float)  # the global local imbalance based weight
        for i in range(self._num_sims):
            S1 = Ss[i] - np.diag(np.diag(Ss[i])) # set diagnol elements to zeros
            _, C = self._cal_limb(S1, Y, self.k)
            # self.wg[i] = 1- milb
            
            idx1 = np.where(Y==1)
            C[idx1] = 1-C[idx1]
            W[i,:] = np.sum(C, axis=1) #+1.0/(self.k*self._num_sims)
            # W[i,h] is the easiness of d_h in i-th Sim, 1.0/self.k is an smoothing parameter ensuring none zero in W
        
        # set smaller rn_sims weights in each column of W to zero
        # if self.rn_sims>0:
        #     idx_par = np.argpartition(self.wg, kth=self.rn_sims)
        #     self.wg[idx_par[:self.rn_sims]] = 0
        # self.sum_wg = np.sum(self.wg)
            
        wg = np.sum(W, axis=1)
        sum_W_rows = np.sum(W, axis=0)  
        idx0 = np.where(sum_W_rows==0)[0] # indices of durgs whose sum of weight is zero
        # sum_W_rows[idx0] = self.sum_wg
        W[:,idx0] = wg[:,None]
        
        # set smaller rn_sims weights in each column of W to zero
        if self.rn_sims>0:
            idx_par = np.argpartition(W, kth=self.rn_sims, axis=0)
            W[idx_par[:self.rn_sims,:],np.arange(W.shape[1])[None,:]] = 0   
            
        sum_W_rows = np.sum(W, axis=0) # recompute the sum of each columns   
        sum_W_rows[sum_W_rows==0] = 1
        W = W/sum_W_rows[None,:] # the sum of each columns in W is 1
        return W
    #---------------------------------------------------------------------------------------- 
    
    def _cal_limb(self, S, Y, k):
        """ S is similarity matrix whose dignoal elememets are zeros"""
        
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        neigh.fit(np.zeros(S.shape))
        knns = neigh.kneighbors(1 - S, return_distance=False)
        
        C = np.zeros(Y.shape, dtype=float)
        for i in range(Y.shape[0]):
            ii = knns[i]
            s = S[i,ii]
            z = np.sum(s)
            if z == 0:
                z=1
            C[i] = 1-s@Y[ii,:]/z
        C *= Y #
        milb = np.sum(C)/np.sum(Y)
        
        return milb, C
    #---------------------------------------------------------------------------------------- 
    
#---------------------------------------------------------------------------------------- 



    
    
class Combine_Sims_LimbPerDT2(Combine_Sims_LimbPerDT):
    """ Difference with Combine_Sims_LimbPerDT: considering the influence of similarities in '_cal_limb' function
    Seems slightly good for MFAP but worse for MFAUC, compared with Combine_Sims_Limb3
    """
    
    def combine_test(self, Ss_te): # No change
        # combine the test similarities, Ss is the test similairties
        self._check_test_sim(Ss_te)
        self._m = Ss_te.shape[1] # the number of test drugs/targets
        neigh = NearestNeighbors(n_neighbors=self.k, metric='precomputed')
        neigh.fit(np.zeros((self._n,self._n)))
        W_te = np.zeros((self._num_sims,self._m), dtype=float) 
        
        for i in range(self._num_sims):
            knn = neigh.kneighbors(1 - Ss_te[i], return_distance=False) # knn.shape = (m,k)
            U = self.W[i,:][knn] # U.shape = (m,k)
            W_te[i,:] = np.mean(U, axis=1)
        
        sum_W_rows = np.sum(W_te, axis=0)
        W_te = W_te/sum_W_rows[None,:] # the sum of each columns in W is 1
        S_te = self._combine_sim(W_te, Ss_te)        
        return S_te, W_te
    #----------------------------------------------------------------------------------------     
    
    def _compute_weights(self, Ss, Y):  # No change
        W = np.zeros((self._num_sims,self._n), dtype=float) 
        for i in range(self._num_sims):
            S1 = Ss[i] - np.diag(np.diag(Ss[i])) # set diagnol elements to zeros
            C = self._cal_limb(S1, Y, self.k)
            W[i,:] = np.sum(C, axis=1)+1.0/self.k 
            # W[i,h] is the difficulty of d_h in i-th Sim, 1.0/self.k is an smoothing parameter ensuring none zero in W
      
        W = 1/W
        sum_W_rows = np.sum(W, axis=0)
        W = W/sum_W_rows[None,:] # the sum of each columns in W is 1
        return W
    #---------------------------------------------------------------------------------------- 
    
    def _cal_limb(self, S, Y, k):
        """ S is similarity matrix whose dignoal elememets are zeros"""
        
        neigh = NearestNeighbors(n_neighbors=k, metric='precomputed')
        neigh.fit(np.zeros(S.shape))
        knns = neigh.kneighbors(1 - S, return_distance=False)
        
        C = np.zeros(Y.shape, dtype=float)
        for i in range(Y.shape[0]):
            ii = knns[i]
            s = S[i,ii]
            z = np.sum(s)
            if z == 0:
                z=1
            C[i] = 1-s@Y[ii,:]/z
        C *= Y #
        #milb = np.sum(C)/np.sum(Y)
        
        return C
    #---------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------- 