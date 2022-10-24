import numpy as np
import scipy.sparse as sp
from math import exp
from sklearn.neighbors import NearestNeighbors
from base.inbase import InductiveModelBase   


class BRDTI(InductiveModelBase):
    """
    Implementation of BRDTI referring https://github.com/lpeska/BRDTI/blob/master/brdti.py: 
    Peska, Ladislav, Krisztian Buza, and Júlia Koller. "Drug-target interaction prediction: A Bayesian ranking approach." 
        Computer methods and programs in biomedicine 152 (2017): 15-21.    
    """            

    def __init__(self, K=5, num_factors=50, theta=0.1, lambda_r=0.1, lambda_c=0.1, max_iter=100, seed=0):
        self.K = K
        self.num_factors = num_factors # letent feature of drug and target
        self.theta = theta  # learning rate
        self.lambda_r = lambda_r
        self.lambda_c = lambda_c
        self.max_iter = max_iter
        self.seed = seed
        
        self.copyable_attrs = ['K','num_factors','theta','lambda_r','lambda_c','max_iter','seed']  
        #----------------------------------------------------------------------------------------
        
        
    def _get_nearest_neighbors(self, S, size=5):
        """ Eq.8, Eq.9, the S is the similarity matrix whose diagonal elements are 0"""
        m, n = S.shape
        X = np.zeros((m, n))
        neigh = NearestNeighbors(n_neighbors=size, metric='precomputed')
        neigh.fit(np.zeros((m,n)))
        knn_indices = neigh.kneighbors(1-S, return_distance=False) # 1-S is the distance matrix whose diagonal elements are 0
        for i in range(m):
            ii = knn_indices[i]
            X[i, ii] = S[i, ii]
        return X   
    
 
    def fit(self, intMat, drugMat, targetMat, cvs=2):
        self._check_fit_data(intMat, drugMat, targetMat, cvs)

        self.current_theta = self.theta
        self.lambda_rc = self.lambda_r*self.lambda_c
        
        self.data = sp.csr_matrix(intMat)
                
        x, y = np.where(intMat > 0)
        self.train_drugs, self.train_targets = set(x.tolist()), set(y.tolist())
        
        self._prng = np.random.RandomState(self.seed)    
        
        Sd = drugMat - np.diag(np.diag(drugMat))  # dsMat is the drugMat which sets the diagonal elements to 0 
        St = targetMat - np.diag(np.diag(targetMat))
                      
        Sd = self._get_nearest_neighbors(Sd)
        St = self._get_nearest_neighbors(St)
        
        self.Sd_s = (Sd.sum() / Sd.shape[0])   
        self.Sd = (1/self.Sd_s) * Sd
        self.St_s = (St.sum() / St.shape[0])   
        self.St = (1/self.St_s) * St   

        self._Sd_sumrow = np.sum(self.Sd, axis=1)
        self._St_sumrow = np.sum(self.St, axis=1)
        
        self.train()
        
            
    #----------------------------------------------------------------------------------------
    
    
    def train(self):
        self._bias_u = np.zeros(self._n_drugs)
        self._bias_v = np.zeros(self._n_targets)
        self._U = np.sqrt(1/float(self.num_factors)) * self._prng.normal(size=(self._n_drugs,self.num_factors))
        self._V = np.sqrt(1/float(self.num_factors)) * self._prng.normal(size=(self._n_targets,self.num_factors))
               
        num_loss_samples = int(100*self._n_drugs**0.5)
        users, pos_items, neg_items = self._uniform_user_sampling( num_loss_samples)
        self.loss_samples = list(zip(users, pos_items, neg_items)) 
        
        act_loss = self.loss()
        n_samples = self.data.nnz # number of postive labels
        
        for it in range(self.max_iter):
            users, pos_items, neg_items = self._uniform_user_sampling(n_samples)
            for u,i,j in list(zip(users, pos_items, neg_items)):
                self.update_factors(u,i,j)

            #execute bold driver learning  after each epoch  
            new_loss =  self.loss()
            # print(new_loss)
            if new_loss < act_loss:
                self.current_theta = self.current_theta * 1.1
            else:
                self.current_theta = self.current_theta * 0.5
            act_loss = new_loss
    #----------------------------------------------------------------------------------------


    def update_factors(self,u,i,j):
        """apply SGD update"""        
        x = self._bias_v[i] - self._bias_v[j] + np.dot(self._U[u,:],self._V[i,:]-self._V[j,:])
        
        if x > 200:
            z = 0
        if x < -200:
            z = 1
        else:    
            ex = exp(-x)
            z = ex/(1.0 + ex)
         
        # _bias_v[i] 
        d = z - self.lambda_r * self._bias_v[i]
        self._bias_v[i] += self.current_theta * d
        
        # _bias_v[j] 
        d = -z - self.lambda_r * self._bias_v[j]
        self._bias_v[j] += self.current_theta * d  
           
        # _U[u,:]                          
        d = (self._V[i,:]-self._V[j,:])*z - self.lambda_r*self._U[u,:] 
        if self.lambda_rc > 0:
            #code for updating content alingment - based on similarity matrix               
            alignmentVectorU = np.dot(self.Sd[u,:], self._U)   
            alignmentSumU = self._Sd_sumrow[u] 
            d = d + 2*self.lambda_rc * (alignmentVectorU - (alignmentSumU * self._U[u,:]) )
        self._U[u,:] += self.current_theta * d 

        # _V[i,:]                                       
        d = self._U[u,:]*z - self.lambda_r*self._V[i,:] 
        if self.lambda_rc > 0:
            #code for updating content alingment - based on similarity matrix               
            alignmentVectorI = np.dot(self.St[i,:], self._V)
            alignmentSumI = self._St_sumrow[i] 
            d = d + 2*self.lambda_rc * (alignmentVectorI - (alignmentSumI * self._V[i,:]))
        self._V[i,:] += self.current_theta * d 

        # _V[j,:]                              
        d = -self._U[u,:]*z - self.lambda_r*self._V[j,:]
        if self.lambda_rc > 0:
            #code for updating content alingment - based on similarity matrix               
            alignmentVectorJ = np.dot(self.St[j,:], self._V) 
            alignmentSumJ = self._St_sumrow[j] 
            d = d + 2*self.lambda_rc * (alignmentVectorJ - (alignmentSumJ * self._V[j,:]))   
        self._V[j,:] += self.current_theta * d 
    #----------------------------------------------------------------------------------------
        
    def _uniform_user_sampling(self, n_samples):
        """
          Creates `n_samples` random samples from training data for performing Stochastic
          Gradient Descent. We start by uniformly sampling users, 
          and then sample a positive and a negative item for each 
          user sample.
        """
        sgd_users = self._prng.choice(list(self.train_drugs),size=n_samples)
        sgd_pos_items, sgd_neg_items = [], []
        for sgd_user in sgd_users:
            if(len(self.data[sgd_user].indices)==0):
                continue
            pos_item = self._prng.choice(self.data[sgd_user].indices)
            neg_item = self._prng.choice(list(self.train_targets - set(self.data[sgd_user].indices)))
            sgd_pos_items.append(pos_item)
            sgd_neg_items.append(neg_item)

        return sgd_users, sgd_pos_items, sgd_neg_items        
    #----------------------------------------------------------------------------------------
    
    def loss(self):
        ranking_loss = 0;
        for u,i,j in self.loss_samples:
            x = self.predict_trans(u,j) - self.predict_trans(u,i)
            if x > 200:
                rl = 0
            if x < -200:
                rl = 1
            else:    
                ex = exp(-x)
                rl = 1.0/(1.0+ex)
            ranking_loss += rl
                
        complexity = self.complexity()
        return ranking_loss + complexity
    #----------------------------------------------------------------------------------------
    
    def complexity(self):
        complexity = 0
        for u,i,j in self.loss_samples:
            complexity += self.lambda_r * np.dot(self._U[u],self._U[u])
            complexity += self.lambda_r * np.dot(self._V[i],self._V[i])
            complexity += self.lambda_r * np.dot(self._V[j],self._V[j])
            
            complexity += -(self.lambda_rc * np.dot(np.dot(self._V,self._V[i,:]), self.St[:,i]) )
            complexity += -(self.lambda_rc * np.dot(np.dot(self._V,self._V[j,:]), self.St[:,j]) )
            complexity += -(self.lambda_rc * np.dot(np.dot(self._U,self._U[u,:]), self.Sd[:,u]) )
            
            complexity += self.lambda_r * self._bias_v[i]**2
            complexity += self.lambda_r * self._bias_v[j]**2
               
        return complexity
    #----------------------------------------------------------------------------------------
    
    
    def predict_trans(self,u,i):
        if (u not in self.train_drugs) & (i in self.train_targets):
            z = np.sum(self.Sd[u,:]) 
            if z ==0:
                z = 1
            vector_u = self.Sd[u,:]@self._U/z
            ub = np.mean(self._bias_u)
            vector_v = self._V[i,:]
            vb = self._bias_v[i]
        elif (i not in self.train_targets) & (u in self.train_drugs):            
            vector_u = self._U[u,:]
            ub =self._bias_u[u]
            z = np.sum(self.St[i,:]) 
            if z==0:
                z = 1
            vector_v = self.St[i,:]@self._V/z
            vb = np.mean(self._bias_v)
        elif (i not in self.train_targets) & (u not in self.train_drugs):
            z = np.sum(self.Sd[u,:]) 
            if z ==0:
                z = 1
            vector_u = self.Sd[u,:]@self._U/z
            ub = np.mean(self._bias_u)
            z = np.sum(self.St[i,:]) 
            if z==0:
                z = 1
            vector_v = self.St[i,:]@self._V/z
            vb = np.mean(self._bias_v)
        else:
            vector_u = self._U[u,:]
            ub = self._bias_u[u]
            vector_v = self._V[i,:]
            vb = self._bias_v[i]
        
        return vb + ub + np.dot(vector_u,vector_v)
    #----------------------------------------------------------------------------------------
    
    def predict(self, drugMatTe, targetMatTe):
        self._check_test_data(drugMatTe, targetMatTe)        
        scores=np.zeros((self._n_drugs_te,self._n_targets_te),dtype=float)
              
        bu_mean = self._bias_u.mean()
        bv_mean = self._bias_v.mean()
        
        # update U and V corresponding to all-zero rows and columns in interaction matrix, respectively 
        if self._cvs == 3:
            for u in range(self._n_drugs):
                if u not in self.train_drugs:
                    z = np.sum(self.Sd[u,:])
                    if z == 0:
                        z = 1
                    self._U[u,:] = self.Sd[u,:]@self._U/z
                    self._bias_u[u] = bu_mean
        if self._cvs == 2:           
            for v in range(self._n_targets):
                if v not in self.train_targets:
                    z = np.sum(self.St[v,:]) 
                    if z == 0:
                        z = 1
                    self._V[v,:] = self.St[v,:]@self._V/z
                    self._bias_v[v] = bv_mean
        
        drug_dis_te = 1 - drugMatTe
        target_dis_te = 1 - targetMatTe
        if self._cvs == 2 or self._cvs == 4:
            neigh_d = NearestNeighbors(n_neighbors=self.K, metric='precomputed')
            neigh_d.fit(np.zeros((self._n_drugs,self._n_drugs)))
            knn_d = neigh_d.kneighbors(drug_dis_te, return_distance=False)
            
            Sd_te = np.zeros(drugMatTe.shape)
            U_te = np.zeros((self._n_drugs_te, self.num_factors))
            for i in range(self._n_drugs_te): 
                Sd_te[i,knn_d[i]] = drugMatTe[i,knn_d[i]]
                z = np.sum(Sd_te[i,:])
                if z == 0:
                    z = 1
                U_te[i,:] = Sd_te[i,:]@self._U/z
                
        if self._cvs == 3 or self._cvs == 4:  
            neigh_t = NearestNeighbors(n_neighbors=self.K, metric='precomputed')
            neigh_t.fit(np.zeros((self._n_targets,self._n_targets)))
            knn_t = neigh_t.kneighbors(target_dis_te, return_distance=False)
            
            St_te = np.zeros(targetMatTe.shape)
            V_te = np.zeros((self._n_targets_te, self.num_factors))
            for i in range(self._n_targets_te): 
                St_te[i,knn_t[i]] = targetMatTe[i,knn_t[i]]
                z = np.sum(St_te[i,:]) 
                if z == 0:
                    z = 1
                V_te[i,:] = St_te[i,:]@self._V/z

        if self._cvs == 2:
            V_te = self._V
            scores = U_te@V_te.T
            scores += bu_mean + self._bias_v
        elif self._cvs==3:
            U_te = self._U
            scores = U_te@V_te.T
            scores += self._bias_u[:,None] + bv_mean
        elif self._cvs==4:
            scores = U_te@V_te.T
            scores += bu_mean + bv_mean
        
        # for d in range(self._n_drugs_te):
        #     for t in range(self._n_targets_te):
        #         if self._cvs==3: # known drug
        #             u=self._U[d,:]
        #             bu = self._bias_u[d]
        #         else: # new drug S2, S4
        #             ii = knn_d[d]
        #             u= np.dot(drugMatTe[d, ii], self._U[ii, :])/np.sum(drugMatTe[d, ii])
        #             bu = bu_mean
        #         if self._cvs==2:   # known target
        #             v=self._V[t,:]
        #             bv = self._bias_v[t]
        #         else:  # new target S3,S4
        #             jj = knn_t[t]  
        #             v= np.dot(targetMatTe[t, jj], self._V[jj, :])/np.sum(targetMatTe[t, jj])  
        #             bv = bv_mean
        #         val=np.sum(u*v)+bu+bv
        #         scores[d,t]=val
        sc = np.exp(scores)
        sc = sc/(1+sc)
        sc[scores>200] = 1
        sc[scores<-200]= 0
        scores = sc
        return scores
    #----------------------------------------------------------------------------------------
    
    def __str__(self):
        return "Model: BRDTI, factors:%s, learningRate:%s,  max_iters:%s, lambda_r:%s, lambda_c:%s, simple_predict:%s" % (self.num_factors, self.theta, self.max_iter, self.lambda_r, self.lambda_c)
    



