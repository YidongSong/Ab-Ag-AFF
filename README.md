# Introduction
![figure1](https://github.com/YidongSong/Ab-Ag-AFF/blob/main/Figs/Model%20architecture.png)

The model extracts sequence features from the heavy and light chains of antibodies using an antibody language model, and simultaneously captures rich evolutionary information from the antigen sequence via ESM2. These integrated features are then fed into a deep learning framework to enable accurate prediction of antibody-antigen binding affinity.

# System requirement
The model is developed under Linux environment with:
Python 3.8.16, pyg v2.3.0, numpy v1.24.3, pytorch v1.13.1, biopython v1.83, decorator v5.1.1, scipy 1.10.1, filelock, v3.12.1, gmp v6.2.1, idna v3.4, ipython v8.12.0, openfold v1.0.1, and six v1.16.0

# Install and run the model
**1.** Clone this repository by 'git clone https://github.com/YidongSong/Ab-Ag-AFF.git'.

**2.** Install the aforementioned packages.
```
conda create -n <env_name> python==3.8
conda activate <env_name>
conda install <the aforementioned packages>
```

**3.** Download the models.
The pre-trained model weights were obtained from https://zenodo.org/records/16974510. Specifically, ```ESM_models.tar.gz``` was decompressed into the directory ```./esm2```, while ```Ab_Ag_AFF.tar.gz```, ```antibody_tokenizer.tar.gz```, ```heavy.tar.gz```, and ```light.tar.gz``` were extracted to ```./model``` to ensure proper model loading and inference.

**4.** Configure your file to be predicted in the JSON file.

Open ``` config/common/bert_eval_generation.json ``` and update the `data_dir` parameter to specify the path of the input data file intended for prediction.

**5.** Run the model with the following command:  
```
python predict.py
```

The prediction results will be stored in the ```./Results``` directory, with subdirectories named according to the timestamp of the computation.

**6.** Analysis of prediction results.

| Heavy chain          | Light chain                         | Antigen                     |Predictions      |
|:-------------------:|:-------------------------------:|:------------------------------:|:------------------------------:|
|QVQ...GLE...VSS| QSV...DRF...TVL|RVQPTE......NKCVNF|0.4819|
|QVQ...WMG...VSS|QSV...SGV...TVL|RVQPTE......NKCVNF|0.4413|
|QVQ...SCK...VSS|QSV...TAP...TVL|RVQPTE......NKCVNF|0.8461|

The ```Predictions``` represent the predicted ```log₁₀IC₅₀``` values, where lower values indicate stronger binding affinity.

# Citation and contact
Citation: preparing

In case you have questions, please contact Yidong Song (songyd6@mail2.sysu.edu.cn).  

