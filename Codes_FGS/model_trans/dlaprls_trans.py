import numpy as np
from sklearn.neighbors import NearestNeighbors
from base.transbase import TransductiveModelBase

class DLapRLS_TRANS(TransductiveModelBase):
    """ 
    Impelementation of DLapRLS based on
    matlab codes of [1] which is availble at https://figshare.com/s/f664b36119c60e7f6f30
    [1] Ding, Yijie, Jijun Tang, and Fei Guo. "Identification of drug–target interactions via 
    dual laplacian regularized least squares with multiple kernel fusion." Knowledge-Based Systems 204 (2020): 106254.
    """
    
    def __init__(self, lambda_d=0.9, lambda_t=0.9, u=1.0, v=1.0, max_iter=10, seed=0, usewknkn=True):
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.u = u # 0<u<2 
        # self.v = v # u+v = 2
        self.max_iter = max_iter
        self.seed = seed
        self.usewknkn = usewknkn
        
        self.copyable_attrs = ['lambda_d', 'lambda_t', 'max_iter', 'u', 'seed', 'usewknkn']   
        """
        lambda_d = 0.9 or [0.9,0.95,1.0]
        lambda_t = 0.9 or [0.9,0.95,1.0]
        """
    #----------------------------------------------------------------------------------------
    
    def fit(self, intMat, drugMat, targetMat, test_indices, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, test_indices, cvs)
        
        self.v = 2-self.u
        if self.usewknkn:
            Y = self._wknkn(intMat, drugMat, targetMat, test_indices)
        else:
            Y = intMat
        # compute normalized Laplacian matrice Ld and Lt
        Ld = self._norm_laplacian_matrix(drugMat) # eq.2a
        Lt = self._norm_laplacian_matrix(targetMat) # eq.2b
        
        prng = np.random.RandomState(self.seed)
        Ad = prng.normal(size=(self._n_drugs,self._n_targets))
        At = prng.normal(size=(self._n_targets,self._n_drugs))
        for i in range(self.max_iter):
            Ad1 = self.u*drugMat@drugMat+self.lambda_d*Ld
            Ad2 = 2*Y-self.v*(At.T)@(targetMat.T)
            try:
                Ad = np.linalg.inv(Ad1)@drugMat@Ad2  
            except np.linalg.LinAlgError:
                Ad = np.linalg.pinv(Ad1)@drugMat@Ad2  
        
            At1 = self.v*targetMat@targetMat+self.lambda_t*Lt
            At2 = 2*Y.T-self.u*(Ad.T)@(drugMat.T)
            try:
                At = np.linalg.inv(At1)@targetMat@At2
            except np.linalg.LinAlgError:
                At = np.linalg.pinv(At1)@targetMat@At2
            
        scores = (self.u*drugMat@Ad + self.v*(targetMat@At).T)/2
        S_te = self._get_test_scores(scores)
        S_te[S_te==np.inf] = 0.0
        S_te[np.isnan(S_te)] = 0.0
        return S_te
    #----------------------------------------------------------------------------------------

    def _norm_laplacian_matrix(self, S):
        d = np.sum(S, axis=1)
        D = np.diag(d)
        Delt = D - S
        d1 = d**-0.5 # d1 =d^(-1/2)
        d1[d1==np.inf] = 0.0
        D1 = np.diag(d1)  # D1 = D^(-1/2)       
        L = D1@Delt@D1
        return L 


    def _wknkn(self, Y, Sd, St, test_indices, K=3, T=0.5):
        if self._cvs == 1:
            Yd = self._recover_intMat(Y, Sd, [], K, T)
            Yt = self._recover_intMat(Y.T, St, [], K, T)
        elif self._cvs == 2:
            test_d = test_indices
            Yd = self._recover_intMat(Y, Sd, test_d, K, T)
            Yt = self._recover_intMat(Y.T, St, [], K, T)
        elif self._cvs == 3:
            test_t = test_indices
            Yd = self._recover_intMat(Y, Sd, [], K, T)
            Yt = self._recover_intMat(Y.T, St, test_t, K, T)
        elif self._cvs == 4:
            test_d,test_t = test_indices
            Yd = self._recover_intMat(Y, Sd, test_d, K, T)
            Yt = self._recover_intMat(Y.T, St, test_t, K, T)
            
        Ydt = (Yd+Yt.T)/2
        Y_new = np.maximum(Y, Ydt)
        return Y_new
        
    
    def _recover_intMat(self, Y, S, test_d, K=3, T=0.5):
        n = S.shape[0] #  number of drugs, the number of rows in Y
        Y_new = np.zeros(Y.shape, dtype=float)
        if len(test_d) == 0: # all drugs are in traning set if test_d=[], 
            S_te = S
            Y_tr = Y
        else:
            all_d = np.arange(n, dtype=int)
            train_d = np.setdiff1d(all_d,test_d) # training drug indices 
            S_te = S[:,train_d] # columns for training drugs
            Y_tr = Y[train_d,:] # only contain training drugs interactions
        neigh_d = NearestNeighbors(n_neighbors=K, metric='precomputed')
        neigh_d.fit(np.zeros((S_te.shape[1],S_te.shape[1])))
        S_te[S_te>1] = 1.0
        knn_d = neigh_d.kneighbors(1.0-S_te, return_distance=False) # only contain knn of test drugs    
        
        w = np.arange(K, dtype=int)
        w = T**w
        for d in range(n): 
            ii = knn_d[d]
            Y_new[d,:]=(w*S_te[d,ii])@Y_tr[ii, :]
            z = np.sum(S_te[d,ii])
            if z == 0:
                z = 1
            Y_new[d,:]/=z
        return Y_new
       
        
    #----------------------------------------------------------------------------------------   

    def __str__(self):
        return "Model: DLapRLS_TRANS, lambda_d:%s, lambda_t:%s, u:%s, v:%s, max_iter:%s, usewknkn:%s, " % (self.lambda_d, self.lambda_t, self.u, self.v, self.max_iter, self.usewknkn)


    
        
    
