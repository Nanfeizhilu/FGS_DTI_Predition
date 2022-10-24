import os
import time
import sys
import numpy as np
from sklearn.metrics import jaccard_score


from base.crossvalidation import *
from base.loaddata import * 


from mf.nrlmf import NRLMF
from nn.wknknri4 import WKNKNRI4_
from nn.wknkn import WKNKN
from mf.brdti import BRDTI
from mf.grmf import *


from model_trans.mf.nrlmf_trans import NRLMF_TRANS
from model_trans.mf.brdti_trans import BRDTI_TRANS
from model_trans.mf.grmf_trans import GRMF_TRANS
from model_trans.dlaprls_trans import DLapRLS_TRANS
from model_trans.wknknri4__trans import WKNKNRI4__TRANS 

from mv.com_sim.combine_sims import *
from mv.com_sim.combine_sims2 import *
from mv.com_sim.hsic import HSIC
from mv.com_sim.cs_snf import Combine_Sims_SNF
from mv.mv_model.lcs_sv import LCS_SV,LCS_SV2


from mv.mv_model_trans.lcs_sv_trans import LCS_SV_TRANS,LCS_SV2_TRANS

"""
Similarity Integration methods:
    method name in codes        method  Ref
    Combine_Sims_Ave            AVE     [1]     
    Combine_Sims_KA             KA      [2] 
    Combine_Sims_Limb3          LIC     [3] 
    HISC                        HSIC    [4]
    Combine_Sims_SNF            SNF     [5]
Notes: Combine_Sims_SNF with method = 'DDR' in load_datasets function is SNF-H [6]
       Combine_Sims_SNF with method = 'FSS' in load_datasets function is SNF-F [7]
        
    Combine_Sims_LimbPerDT_2    FGS     propsoed 

Base models:
    method name in codes            method  Ref
    WKNKNRI4_ & WKNKNRI4_TRANS      WkNNIR  [8]
    NRLMF & NRLMF_TRANS             NRLMF   [9]
    GRMF_WKNKN_TRANS      GRMF    [10]
    BRDTI & BRDTI_TRANS             BRDTI   [11]
    DLapRLS_TRANS                   DLapRLS  [4]
"""

 

def initialize_model_mv(method, cvs=2):
    if method == 'LCS_SV':
        model = LCS_SV(cs_model=Combine_Sims_Ave(), sv_model=WKNKN(k=7, T=0.8))
    elif method == 'LCS_SV_TRANS':
        model = LCS_SV_TRANS(cs_model=Combine_Sims_Ave(), sv_model=NRLMF_TRANS(cfix=5, K1=5, K2=5, num_factors=50, theta=1.0, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0))
    
    elif method == 'LCS_SV2':
        model = LCS_SV2(cs_model=Combine_Sims_LimbPerDT(k=5), sv_model=WKNKN(k=7, T=0.8))
    elif method == 'LCS_SV2_TRANS':
        model = LCS_SV2_TRANS(cs_model=Combine_Sims_LimbPerDT(k=5), sv_model=WKNKNRI4__TRANS(k=7, T=0.8))

    else:
        raise RuntimeError("The method name: {} has not been defined initialize_model_mv function!!".format(method))
    return model
#----------------------------------------------------------------------------------------
        
def initialize_model_cs(method, cvs =2):
    if method == 'Combine_Sims_Ave':
        model = Combine_Sims_Ave()
    elif method == 'Combine_Sims_KA':
        model = Combine_Sims_KA()
    elif method == 'Combine_Sims_Limb3': # Combine_Sims_Limb3 is LIC
        model = Combine_Sims_Limb3(k = 5)
    
    
    elif method == 'Combine_Sims_LimbPerDT_2':
        model = Combine_Sims_LimbPerDT_2(k=5, rho=0.5) # rho=0.6, 0.2, 0.7
    
    
    elif method == 'HSIC':
        model = HSIC(v1=2**-1, v2=2**-4, seed=0) # v1=2**-1, v2=2**-4
    elif method == 'Combine_Sims_SNF':
        model = Combine_Sims_SNF(k=3, num_iters=2, alpha=1)
    else:
        raise RuntimeError("The method name: {} has not been defined initialize_model_cs function!!".format(method))
    return model    
#----------------------------------------------------------------------------------------

def initialize_model_sv(method, cvs=2):
    model = None
    if method == 'NRLMF':
        model = NRLMF(cfix=5, K1=5, K2=5, num_factors=50, theta=0.1, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0)
    elif method == 'WKNKNRI4_':
        model = WKNKNRI4_(k=7, kr=7, T=0.8)    
    elif method == 'BRDTI':
        model = BRDTI(K=5, num_factors=50, theta=0.1, lambda_r=0.01, lambda_c=1.0, max_iter=100, seed=0)
    
    elif method == 'NRLMF_TRANS':
        model = NRLMF_TRANS(cfix=5, K1=5, K2=5, num_factors=50, theta=0.1, lambda_d=0.25, lambda_t=0.25, alpha=0.25, beta=0.25, max_iter=100, seed=0)
    elif method == 'BRDTI_TRANS':
        model = BRDTI_TRANS(K=5, num_factors=50, theta=0.1, lambda_r=0.01, lambda_c=1.0, max_iter=100, seed=0)                         
    elif method == 'GRMF_WKNKN_TRANS':
        a = 0.25
        model = GRMF_TRANS(K=5, T=0.7, num_factors=50, lambda_d=a, lambda_t=a, lambda_r=a, max_iter=5, seed=0, is_wknkn=1)
    elif method == 'DLapRLS_TRANS':
        model = DLapRLS_TRANS(lambda_d=0.9, lambda_t=0.9, u=1.0, v=1.0, max_iter=10, seed=0, usewknkn=True)
    elif method == 'WKNKNRI4__TRANS':
        model = WKNKNRI4__TRANS(k=7, kr=7, T=0.8) 
    
 
    else:
        raise RuntimeError("The method name: {} has not been defined initialize_model_sv function!!".format(method))
    return model
#----------------------------------------------------------------------------------------
    


if __name__ == "__main__":
    # !!! my_path should be change to the path of the project in your machine  
    isHyperion = True  # False True       
    isInductive = True
    metric = 'auc_aupr' #'auc_aupr' #'auc' 'aupr' 'auc_aupr'     
    if isHyperion: # run in Hyperion
        my_path = '/WCBD_5_tmp/GitHub_MF2A_MDMF2A'     # /data/mlkd/BinLiu/DPI_Py37  # set the path of 'LB_code' file to 'my_path'  
        n_jobs = 4 # set the n_jobs = 20 if possible
    else: # run in my labtop
        my_path = 'F:\envs\GitHub_MF2A_MDMF2A\Codes_FGS'   
        n_jobs = 5 #5

    data_dir =  os.path.join(my_path, 'datasets_mv') #'F:\envs\DPI_Py37\Codes_Py3\datasets' #'data'
    output_dir = os.path.join(my_path, 'output', 'Results')   # 'HISC best_param_LimbPerDT_2 VP'  'Combine_Sims_LimbPerDT_2 VP BRDTI'
    # 'Combine_Sims_LimbPerDT_2 VP WKNKNRI4_'
    # 'SNF FSS best_param_LimbPerDT_2 WKNKNRI4_' # DDR FSS
    # 'other combine_sims allsims best_param_LimbPerDT_2 WKNKNRI4_'
    #'F:\envs\DPI_Py37\Codes_Py3\output' 
    seeds = [0,1] #[7] 
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir) 
    out_summary_file = os.path.join(output_dir, "summary_result"+ time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime()) +".txt")

    parameter_setting = 1 # 1,2,3 
    
    # prpopsoed FGS
    if parameter_setting == 1:
        """ run cross validation on a model with best parameters setting from file"""
        cmd ="Method\tsv_method\tcs_method\tcvs\tDataSet\tAUPR\tAUC\tTime\tTotalTime\tparam"
        cs_method_param = 'Combine_Sims_LimbPerDT' # 'Combine_Sims_Limb3' # the cs_method for getting parameters
        # # the best param for Combine_Sims_Limb3 are used to other combine_similarity methods (Combine_Sims_Ave, Combine_Sims_KA...)
        print(cmd)
        with open(out_summary_file,"w") as f:
            f.write(cmd+"\n")
        for sv_method_ori in ['WKNKNRI4_','NRLMF', 'GRMF_WKNKN_TRANS', 'BRDTI', 'DLapRLS_TRANS' ]: # 'NRLMF','BRDTI' 'GRMF_WKNKN_TRANS','DLapRLS_TRANS' 'KronSVM_TRANS'   'WKNKNRI4_' 'GRGMF_TRANS' 
            # 'MF2A_TRANS' 'NRMFAP2_GD_TRANS','NRMFAUC_F3_TRANS' 'NRMFAP2_GD','NRMFAUC_F3'  'WKNKNRI4_','NRLMF', 'KronSVM'
            for cs_method in ['Combine_Sims_LimbPerDT_2']:  
                # 'Combine_Sims_Ave','Combine_Sims_KA','HSIC','Combine_Sims_Limb3'
                # 'Combine_Sims_SNF'
                # 'Combine_Sims_LimbPerDT_2'
                
                for method_ori in ['LCS_SV2']:  # 'LCS_SV', 'LCS_SV2', 'LCS_SV_TRANS', 'LCS_SV2_TRANS', 
                    for cvs in [2,3,4]: # 2,3,4
                        if cvs == 1 and '_TRANS' not in sv_method_ori:
                            method = method_ori+'_TRANS'
                            sv_method = sv_method_ori+'_TRANS'
                        elif '_TRANS' in sv_method_ori:
                            if '_TRANS' not in method:
                                sv_method = sv_method_ori
                                method = method_ori+'_TRANS'
                        else:
                            method = method_ori
                            sv_method = sv_method_ori
                        
                        if '_TRANS' in sv_method:
                            isInductive = False
                        else:
                            isInductive = True
                        
                        full_method =  method+'_'+sv_method+'_'+cs_method # for output 
                        vp_best_param_file = os.path.join(data_dir, 'method_params_VP data1','Test_Combine_Sims_sv_model_best_Param_LimbPerDT_2.txt') # _DSI
                        dict_params = get_params2(vp_best_param_file, num_key=3) # read parameters from file
                        
                    
                        model = initialize_model_mv(method,cvs)  # parammeters could be changed in "initialize_model" function
                        model.sv_model = initialize_model_sv(sv_method, cvs)
                        model.cs_model = initialize_model_cs(cs_method, cvs)
                        num = 10
                        if cvs == 4:
                            num = 3
                        for dataset in ['nr1']:  # 'nr1','gpcr1','ic1','e1','luo'
                            if dataset == 'luo': seeds = [0]
                            else: seeds = [0,1]     
                        
                            if 'Combine_Sims_LimbPerDT_2' in cs_method: 
                                if dataset == 'luo': 
                                    if cvs == 3: model.cs_model.rho=0.7
                                    else: model.cs_model.rho=0.2       
                                else: model.cs_model.rho=0.5
                        
                            out_file_name= os.path.join(output_dir, "Best_parameters_"+full_method+"_"+"S"+str(cvs)+"_"+dataset+".txt") 
                            intMat, drugMats, targetMats, Dsim_names, Tsim_names = load_datasets(dataset, data_dir , method='all') # ,'Original','low4Limb', 'all', 'DDR', 'FSS'
                            
                            
                            param = dict_params[(sv_method.replace("_TRANS", ""), str(cvs), dataset)] # reomve _TRANS suffix
                            model.sv_model.set_params(**param)
                            
                            tic = time.time()
                            # auprs, aucs, run_times = cross_validate(model, cvs, num, intMat, drugMats, targetMats, seeds, out_file_name, isInductive)
                            auprs, aucs, run_times = cross_validate_parallel(model, cvs, num, intMat, drugMats, targetMats, seeds, out_file_name, n_jobs, isInductive)
                            cmd = "{}\t{}\t{}\t{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t".format(method,sv_method,cs_method,cvs,dataset,auprs.mean(),aucs.mean(),run_times.mean(),time.time()-tic)
                            print(cmd)
                            with open(out_summary_file,"a") as f:
                                f.write(cmd+"\n")
    elif parameter_setting == 2: # other linear methods (AVE, KA, HISC and LIC)
        """ run cross validation on a model with best parameters setting from file"""
        cmd ="Method\tsv_method\tcs_method\tcvs\tDataSet\tAUPR\tAUC\tTime\tTotalTime\tparam"
        cs_method_param = 'Combine_Sims_LimbPerDT' # 'Combine_Sims_Limb3' # the cs_method for getting parameters
        # # the best param for Combine_Sims_Limb3 are used to other combine_similarity methods (Combine_Sims_Ave, Combine_Sims_KA...)
        print(cmd)
        with open(out_summary_file,"w") as f:
            f.write(cmd+"\n")
        for sv_method_ori in ['WKNKNRI4_','NRLMF', 'GRMF_WKNKN_TRANS', 'BRDTI', 'DLapRLS_TRANS' ]: # 'NRLMF','BRDTI' 'GRMF_WKNKN_TRANS','DLapRLS_TRANS' 'KronSVM_TRANS'   'WKNKNRI4_' 'GRGMF_TRANS' 
            # 'MF2A_TRANS' 'NRMFAP2_GD_TRANS','NRMFAUC_F3_TRANS' 'NRMFAP2_GD','NRMFAUC_F3'  'WKNKNRI4_','NRLMF', 'KronSVM'
            for cs_method in ['Combine_Sims_Ave','Combine_Sims_KA','HSIC','Combine_Sims_Limb3']:  
                for method_ori in ['LCS_SV']:  # 
                    for cvs in [2,3,4]: # 2,3,4
                        if cvs == 1 and '_TRANS' not in sv_method_ori:
                            method = method_ori+'_TRANS'
                            sv_method = sv_method_ori+'_TRANS'
                        elif '_TRANS' in sv_method_ori:
                            if '_TRANS' not in method_ori:
                                sv_method = sv_method_ori
                                method = method_ori+'_TRANS'
                        else:
                            method = method_ori
                            sv_method = sv_method_ori
                        
                        if '_TRANS' in sv_method:
                            isInductive = False
                        else:
                            isInductive = True
                        
                        full_method =  method+'_'+sv_method+'_'+cs_method # for output 
                        vp_best_param_file = os.path.join(data_dir, 'method_params_VP data1','Test_Combine_Sims_sv_model_best_Param_LimbPerDT_2.txt') # _DSI
                        dict_params = get_params2(vp_best_param_file, num_key=3) # read parameters from file
                        
                    
                        model = initialize_model_mv(method,cvs)  # parammeters could be changed in "initialize_model" function
                        model.sv_model = initialize_model_sv(sv_method, cvs)
                        model.cs_model = initialize_model_cs(cs_method, cvs)
                        num = 10
                        if cvs == 4:
                            num = 3
                        for dataset in ['nr1']:  # 'nr1','gpcr1','ic1','e1','luo'
                            if dataset == 'luo': seeds = [0]
                            else: seeds = [0,1]     
                        
                            if cs_method == 'HSIC':
                                vp_best_param_file = os.path.join(data_dir, 'method_params_VP data1','Test_Combine_Sims_HSIC_best_Param.txt') # best param of HSIC
                                dict_params2 = get_params2(vp_best_param_file, num_key=2) # read parameters from file
                                param = dict_params2[(cs_method.replace("_TRANS", ""), dataset)] # reomve _TRANS suffix
                                model.cs_model.set_params(**param)   
                        
                            out_file_name= os.path.join(output_dir, "Best_parameters_"+full_method+"_"+"S"+str(cvs)+"_"+dataset+".txt") 
                            intMat, drugMats, targetMats, Dsim_names, Tsim_names = load_datasets(dataset, data_dir , method='all') # ,'Original','low4Limb', 'all', 'DDR', 'FSS'
                            
                            
                            param = dict_params[(sv_method.replace("_TRANS", ""), str(cvs), dataset)] # reomve _TRANS suffix
                            model.sv_model.set_params(**param)
                            
                            tic = time.time()
                            # auprs, aucs, run_times = cross_validate(model, cvs, num, intMat, drugMats, targetMats, seeds, out_file_name, isInductive)
                            auprs, aucs, run_times = cross_validate_parallel(model, cvs, num, intMat, drugMats, targetMats, seeds, out_file_name, n_jobs, isInductive)
                            cmd = "{}\t{}\t{}\t{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t".format(method,sv_method,cs_method,cvs,dataset,auprs.mean(),aucs.mean(),run_times.mean(),time.time()-tic)
                            print(cmd)
                            with open(out_summary_file,"a") as f:
                                f.write(cmd+"\n") 

    elif parameter_setting == 3: # other nonlinear methods (SNF-H, SNF-F)
        """ run cross validation on a model with best parameters setting from file"""
        cmd ="Method\tsv_method\tcs_method\tcvs\tDataSet\tAUPR\tAUC\tTime\tTotalTime\tparam"
        cs_method_param = 'Combine_Sims_LimbPerDT' # 'Combine_Sims_Limb3' # the cs_method for getting parameters
        # # the best param for Combine_Sims_Limb3 are used to other combine_similarity methods (Combine_Sims_Ave, Combine_Sims_KA...)
        print(cmd)
        with open(out_summary_file,"w") as f:
            f.write(cmd+"\n")
        for sv_method_ori in ['WKNKNRI4__TRANS','NRLMF_TRANS', 'GRMF_WKNKN_TRANS', 'BRDTI_TRANS', 'DLapRLS_TRANS' ]:  # SNF can only used for the Transductive model and setting
            for cs_method in ['Combine_Sims_SNF']:  
                for method_ori in ['LCS_SV_TRANS']:  # 
                    for cvs in [2,3,4]: # 2,3,4
                        sv_method = sv_method_ori
                        method = method_ori
                        if '_TRANS' in sv_method:
                            isInductive = False
                        else:
                            isInductive = True
                        
                        full_method =  method+'_'+sv_method+'_'+cs_method # for output 
                        vp_best_param_file = os.path.join(data_dir, 'method_params_VP data1','Test_Combine_Sims_sv_model_best_Param_LimbPerDT_2.txt') # _DSI
                        dict_params = get_params2(vp_best_param_file, num_key=3) # read parameters from file
                        
                    
                        model = initialize_model_mv(method,cvs)  # parammeters could be changed in "initialize_model" function
                        model.sv_model = initialize_model_sv(sv_method, cvs)
                        model.cs_model = initialize_model_cs(cs_method, cvs)
                        num = 10
                        if cvs == 4:
                            num = 3
                        for dataset in ['nr1']:  # 'nr1','gpcr1','ic1','e1','luo'
                            if dataset == 'luo': seeds = [0]
                            else: seeds = [0,1]     
                        
                            if cs_method == 'HSIC':
                                vp_best_param_file = os.path.join(data_dir, 'method_params_VP data1','Test_Combine_Sims_HSIC_best_Param.txt') # best param of HSIC
                                dict_params2 = get_params2(vp_best_param_file, num_key=2) # read parameters from file
                                param = dict_params2[(cs_method.replace("_TRANS", ""), dataset)] # reomve _TRANS suffix
                                model.cs_model.set_params(**param)   
                        
                            out_file_name= os.path.join(output_dir, "Best_parameters_"+full_method+"_"+"S"+str(cvs)+"_"+dataset+".txt") 
                            intMat, drugMats, targetMats, Dsim_names, Tsim_names = load_datasets(dataset, data_dir , method='FSS')  # !!!!!!! 'DDR' is SNF-H, 'FSS' is SNF-F
                            
                            
                            param = dict_params[(sv_method.replace("_TRANS", ""), str(cvs), dataset)] # reomve _TRANS suffix
                            model.sv_model.set_params(**param)
                            
                            tic = time.time()
                            # auprs, aucs, run_times = cross_validate(model, cvs, num, intMat, drugMats, targetMats, seeds, out_file_name, isInductive)
                            auprs, aucs, run_times = cross_validate_parallel(model, cvs, num, intMat, drugMats, targetMats, seeds, out_file_name, n_jobs, isInductive)
                            cmd = "{}\t{}\t{}\t{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t".format(method,sv_method,cs_method,cvs,dataset,auprs.mean(),aucs.mean(),run_times.mean(),time.time()-tic)
                            print(cmd)
                            with open(out_summary_file,"a") as f:
                                f.write(cmd+"\n")         
