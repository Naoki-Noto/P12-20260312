# P12-20260312
Note1: Since the pkl files were too large to upload to GitHub, they are stored at the following URL (data_AI: Database D, data_AI2+Human: Database E, data_AI2: Database C, data_Human: Database A, data_Random: Database B, data_zinc_25286/data_zinc_50572: ZINC-based databases): https://drive.google.com/drive/folders/1s7Ai9QQx1iIsX_kTXWBnM9N4DGiWXwoG?usp=sharing

https://drive.google.com/drive/folders/1f5_bBmQqlOF0jW735Kg1B5OP3O7US7MO?usp=sharing

# Table of Contents
Environment: An environment for performing each code is stored in this directory./

OPSdesign2: Python (3.13.5) was used as a language, and used packages were deepchem (2.5.0), gymnasium (1.2.0), matplotlib (3.10.6), numpy (2.1.2), pandas (2.3.2), rdkit (2025.3.5), scikit-learn (1.7.1), stable_baselines3 (2.7.0), tensorflow (2.20.0), torch (2.8.0+cu129), torch-geometric (2.6.1), and transformers (4.56.0)

==========================================================================

ML/

- Frag2OPST5: Code for constructing T5-base SMILES generators and evaluating their performance. Due to file-size limitations, the tokenized datasets used for model training and the resulting trained models are available at the following URL: XX

- GCN_model: Code for constructing a GCN-based reward estimator and preparing graph-structured datasets for model training. Due to file-size limitations, the graph-structured datasets are available at the following URL: XX

- OPSDesigner: Code for a RL-based system to design Organic photosensitizers (OPSs; including photoredox catalysts, triplet sensitizers, and so on). The repository also includes the trained RL models used to generate new molecular structures.

  • 2566fragments_4CzIPN: RL model for generating molecules with properties similar to those of 4CzIPN using a fragment pool consisting of 2,566 fragments (fragment pool A).

  • Realistic_mols_4CzIPN: RL model (Method A) for generating molecules with properties similar to those of 4CzIPN using a fragment pool consisting of 100 fragments (fragment pool B).

  • Realistic_mols_4CzIPN_2envs: RL model with two parallel environments (Method B) for generating molecules with properties similar to those of 4CzIPN using a fragment pool consisting of 100 fragments (fragment pool B).

  • Realistic_mols_4CzIPN_TL: RL model with transfer learning-based parameter initialization (Method C) for generating molecules with properties similar to those of 4CzIPN using a fragment pool consisting of 100 fragments (fragment pool B).

(Rpresentative files in each of the above directory are as follows)

  • GCNcalculator.py: Code for a reward prediction module based on a GCN model.
  
  • T5generator.py: Code for a SMILES generator that takes selected molecular fragments as input and generates molecules composed of those fragments.
  
  • OPSDesignerEnv.py: Code for the RL environment, which integrates T5generator and GCNcalculator.
  
  • learning_XX.ipynb: Code for training the RL models.
  
  • OPSDesign.ipynb: Code for generating candidate molecules using the trained RL models.

# Getting Started with OPSDesigner

To set up the environment for OPSDesigner, execute the following commands using the `OPSdesign2.yml` file located in the Environment folder. If some packages are not installed successfully, please refer to the `OPSdesign2.yml` file and install the missing packages manually using `pip`.
```bash
conda env create -n new_env -f OPSdesign2.yml
conda activate new_env
```
As a representative example (fragment pool A, Method B), run the following scripts:  
- `learning_PPO_2envs.ipynb` — Trains the RL model.
- `OPSdesign.ipynb` — Generates molecular candidates using the trained RL model.

Before running these scripts, organize the files as follows:
```bash
OPSDesigner/
└── Realistic_mols_4CzIPN_2envs/
  ├── GCNmodel/
  │ ├── model_mt_sc.pth
  │ └── scaler_mt.pkl
  │
  ├── source/
  │ └── SMILES_list_100.csv
  │
  ├── RLmodels/         # Directory for trained RL models (required by learning_PPO_2envs.ipynb)
  │
  ├── generated_OPSs/   # Directory for generated molecular candidates (required by OPSdesign.ipynb)
  │
  ├── OPSDesignerEnv.py # RL environment required to run both learning_PPO_2envs.ipynb and OPSdesign.ipynb
  ├── T5generator.py    # T5-based SMILES generator integrated into OPSDesignerEnv.py
  ├── GCNcalculator.py  # GCN-based reward estimator integrated into OPSDesignerEnv.py
  │
  ├── learning_PPO_2envs.ipynb
  └── OPSdesign.ipynb
```

**Customizable component**

#1 The SMILES_list_100.csv file can be replaced with any user-defined molecular fragment pool.

#2 We need to set target property values. In the representative example, we used the HOMO and LUMO energies, S0-S1 excitation energy, S0-S1 oscilator strength, S0-T1 excitation energy, and S1-T1 energy gap of 4CzIPN as the target properties. These target values can be freely modified to design molecules with different target properties. Representative property values for several photosensitizers are listed below:

4CzIPN (N#Cc1c(-n2c3ccccc3c3ccccc32)c(C#N)c(-n2c3ccccc3c3ccccc32)c(-n2c3ccccc3c3ccccc32)c1-n1c2ccccc2c2ccccc21): [-5.9155, -2.7331, 2.5561, 0.0639, 2.4273, 0.1288]

4DPAIPN (N#Cc1c(N(c2ccccc2)c2ccccc2)c(C#N)c(N(c2ccccc2)c2ccccc2)c(N(c2ccccc2)c2ccccc2)c1N(c1ccccc1)c1ccccc1): [-5.3593, -2.2577, 2.4621, 0.086, 2.2258, 0.2363]

PhenS (c1ccc(-c2ccc(-c3ccc4c(c3)Sc3cc(-c5ccc(-c6ccccc6)cc5)ccc3N4c3cccc4ccccc34)cc2)cc1): [-5.1664, -1.6215, 2.9085, 0.0076, 2.5846, 0.3239]
