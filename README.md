# FGS_DTI_Predition
The implemetaion of the paper "Fine-Grained Selective similairty integartion method for DTI prediction", please see [the paper](https://academic.oup.com/bib/advance-article-abstract/doi/10.1093/bib/bbad085/7076223?utm_source=advanceaccess&utm_campaign=bib&utm_medium=email) for more details.

The codes were tested on Python 3.7

## Requirement ##

joblib==0.14.1

numpy==1.17.4

scikit-learn==1.0.1

scipy==1.6.1

## Running Codes ##

Excute the file [main_FGS.py](https://github.com/Nanfeizhilu/FGS_DTI_Predition/blob/main/Codes_FGS/main_FGS.py), with assigning your local directory to the `my_path` variable. The results will be saved in the `output` folder.

In `main_fgs.py`: 

If set `parameter_setting = 1`, the program excutes FGS with five base models (WkNNIR, NRLMF, GRMF, BRDTI and DLapRLS).

If set `parameter_setting = 2`, the program excutes AVE, KA, HSIC, and LIC similarity intergation methods with the five base models.

If set `parameter_setting = 3`, the program excutes two nonlinear methods (SNF-H, SNF-F) with the five base models.

## Contact ##
Bin Liu: [liubin@cqupt.edu.cn](liubin@cqupt.edu.cn)
