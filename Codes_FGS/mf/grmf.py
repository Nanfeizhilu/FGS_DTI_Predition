import numpy as np
from sklearn.neighbors import NearestNeighbors
from base.inbase import InductiveModelBase

class GRMF(InductiveModelBase):
    """
    Implementation of WKNKN+GRMF: 
    Ezzat, Ali, et al. "Drug-target interaction prediction with graph regularized matrix factorization." 
    IEEE/ACM transactions on computational biology and bioinformatics 14.3 (2016): 646-656.
    
    max_iter=2>max_iter=10 for nr and gpcr dataset
    lambda_d(lambda_t) in 2.0**np.arange(-4, 3, 2) is similar with 10.0**np.arange(-4, 1)
    
    The compuation of Laplacians matrix is diffrent with NRLMF:
        1. the sparsified simlarity matrix is symmetric
        2. using normalized Laplacians matrix
    """
    def __init__(self, K=5, T=0.7, num_factors=50, lambda_d=0.25, lambda_t=0.25, lambda_r=0.25, max_iter=100, seed=0, is_wknkn=1):
        self.K = K
        self.T = T
        self.num_factors = num_factors # letent feature of drug and target
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.lambda_r = lambda_r
        self.max_iter = max_iter
        self.seed = seed
        self.is_wknkn = is_wknkn # 1(0): do (not) use wknkn 
        self.copyable_attrs = ['K','T','num_factors','lambda_d','lambda_t','lambda_r','max_iter','seed','is_wknkn']    

    #----------------------------------------------------------------------------------------
        
    def fit(self, intMat, drugMat, targetMat, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, cvs)
        self.acutal_num_factors = min(self.num_factors,self._n_drugs,self._n_targets)
        self._Y = intMat
        
        self._construct_neighborhood(drugMat, targetMat)
        if self.is_wknkn == 1:
            # wknkn preprocess
            Yd = self._recover_intMat(self._knns_d, drugMat, self._Y)
            Yt = self._recover_intMat(self._knns_t, targetMat, self._Y.T)
            Ydt = (Yd + Yt.T)/2
            self._Y = np.maximum(Ydt, self._Y)
        self._initialize()  
        self._optimization()
    #----------------------------------------------------------------------------------------
    
    def _optimization(self):
        I = np.eye(self.acutal_num_factors)
        for i in range(self.max_iter):
            self._U = self._updateU(self._Y, self._U, self._V, self._DL, self.lambda_d, self.lambda_r, I)
            self._V = self._updateU(self._Y.T, self._V, self._U, self._TL, self.lambda_t, self.lambda_r, I)
    #----------------------------------------------------------------------------------------
            
    def _updateU(self, Y, U, V, L, lambda_d, lambda_r, I):
        U1 = Y@V - lambda_d*L@U
        U2 = V.T@V + lambda_r*I
        U2 = np.linalg.inv(U2)
        U = U1@U2
        return U
    #----------------------------------------------------------------------------------------
    
    def _initialize(self):
        u,s,v = np.linalg.svd(self._Y,full_matrices=False)
        s = np.diag(np.power(s,0.5))
        self._U = u@s
        self._U = self._U[:,:self.acutal_num_factors]
        self._V = v.T@s
        self._V = self._V[:,:self.acutal_num_factors] 
    #----------------------------------------------------------------------------------------
    
    def _recover_intMat(self, knns, S, Y):
        Yr = np.zeros(Y.shape)
        etas = self.T**np.arange(self.K)
        for d in range(Y.shape[0]): 
            ii = knns[d]
            sd = S[d,ii]
            z = np.sum(sd)
            if z == 0:
                z = 1
            Yr[d,:] = etas*sd@Y[ii,:]/z
            
        # Yd[Yd>1] = 1
        return Yr
    #----------------------------------------------------------------------------------------
        
    def _construct_neighborhood(self, drugMat, targetMat):
        # construct the laplocian matrices
        dsMat = drugMat - np.diag(np.diag(drugMat))  # dsMat is the drugMat which sets the diagonal elements to 0 
        tsMat = targetMat - np.diag(np.diag(targetMat))
        if self.K > 0:
            S1, self._knns_d = self._get_nearest_neighbors(dsMat, self.K)  # S1 is sparsified durgMat A
            self._DL = self._laplacian_matrix(S1)                   # L^d
            S2, self._knns_t = self._get_nearest_neighbors(tsMat, self.K)  # S2 is sparsified durgMat B
            self._TL = self._laplacian_matrix(S2)                   # L^t
        else:
            self._DL = self._laplacian_matrix(dsMat)
            self._TL = self._laplacian_matrix(tsMat)
    #----------------------------------------------------------------------------------------
            
    def _laplacian_matrix(self, S):
        x = np.sum(S, axis=0)
        L = np.diag(x) - S
        x1 = np.power(x,-0.5)
        x1[x1 == np.inf] = 0
        X1 = np.diag(x1)
        L1 = X1@L@X1 # normalized lapacian matrix 
        return L1
    #----------------------------------------------------------------------------------------
    
    def _get_nearest_neighbors(self, S, size=5):
        """ Eq.9, Eq.10, the S is the similarity matrix whose diagonal elements are 0"""
        m, n = S.shape
        X = np.zeros((m, n))
        neigh = NearestNeighbors(n_neighbors=size, metric='precomputed')
        neigh.fit(np.zeros((m,n)))
        knn_indices = neigh.kneighbors(1-S, return_distance=False) # 1-S is the distance matrix whose diagonal elements are 0
        for i in range(m):
            ii = knn_indices[i]
            X[i, ii] = S[i, ii]
        X = (X+X.T)/2 # difference with NRLMF: the sparsified simlarity matrix is symmetric
        X += np.eye(X.shape[0])
        return X, knn_indices
    #----------------------------------------------------------------------------------------
    
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)        
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
        
        drug_dis_te = 1 - drugMatTe
        target_dis_te = 1 - targetMatTe
        if self._cvs == 2 or self._cvs == 4:
            neigh_d = NearestNeighbors(n_neighbors=self.K, metric='precomputed')
            neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))
            knn_d = neigh_d.kneighbors(drug_dis_te, return_distance=False)
        if self._cvs == 3 or self._cvs == 4:  
            neigh_t = NearestNeighbors(n_neighbors=self.K, metric='precomputed')
            neigh_t.fit(np.zeros((self._n_targets,self._n_targets)))
            knn_t = neigh_t.kneighbors(target_dis_te, return_distance=False)

        U_te = np.zeros((self._n_drugs_te,self.acutal_num_factors), dtype=float)
        V_te = np.zeros((self._n_targets_te,self.acutal_num_factors), dtype=float)
        if self._cvs == 2: 
            for d in range(self._n_drugs_te):
                ii = knn_d[d]
                U_te[d,:]= np.dot(drugMatTe[d, ii], self._U[ii, :])/np.sum(drugMatTe[d, ii])
            V_te = self._V
        elif self._cvs == 3:
            for t in range(self._n_targets_te):
                jj = knn_t[t]
                V_te[t,:]= np.dot(targetMatTe[t, jj], self._V[jj, :])/np.sum(targetMatTe[t, jj])   
            U_te = self._U
        elif self._cvs == 4:
            for d in range(self._n_drugs_te):
                ii = knn_d[d]
                U_te[d,:]= np.dot(drugMatTe[d, ii], self._U[ii, :])/np.sum(drugMatTe[d, ii])
            for t in range(self._n_targets_te):
                jj = knn_t[t]
                V_te[t,:]= np.dot(targetMatTe[t, jj], self._V[jj, :])/np.sum(targetMatTe[t, jj])   
        scores = U_te@V_te.T
    
        # scores[scores==np.inf] = 0.0
        # scores[np.isnan(scores)] = 0.0
        return scores
    #----------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------    
        