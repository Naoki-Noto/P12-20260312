# P12-20260312


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

 
# Setup and Usage
