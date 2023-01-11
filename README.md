# FGS_DTI_Predition
The implemetaion of the Fine-Grained Selective similairty integartion method for DTI prediction, see https://arxiv.org/abs/2212.00543 for more details

## Requirement ##

joblib==0.14.1

numpy==1.17.4

scikit-learn==1.0.1

scipy==1.6.1

>**Requirement**:
joblib==0.14.1
numpy==1.17.4
scikit-learn==1.0.1
scipy==1.6.1

The codes were tested on Python 3.7

Excute the file [Codes/main_fgs.py](https://github.com/Nanfeizhilu/FGS_DTI_Predition/blob/main/Codes_FGS/main_FGS.py), with assigning your local directory to the `my_path` variable. The results are written in the `output` folder.

In `main_fgs.py`: 

If set `parameter_setting = 1`, the program excutes FGS with five base models (WkNNIR, NRLMF, GRMF, BRDTI and DLapRLS).

If set `parameter_setting = 2`, the program excutes AVE, KA, HSIC, and LIC similarity intergation methods with the five base models.

If set `parameter_setting = 3`, the program excutes two nonlinear methods (SNF-H, SNF-F) with the five base models.
