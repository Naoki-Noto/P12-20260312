# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:20:16 2025

@author: noton
"""

import numpy as np
import torchani
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import BFGS
import psi4
import sys
import os
import contextlib

psi4.core.set_output_file(os.devnull, False)
psi4.set_memory('32000 MB')

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def smiles_to_charge(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    charge = Chem.GetFormalCharge(mol)
    return charge


def smiles_to_best_xyz(smiles: str, n_confs: int = 10):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError("Invalid SMILES")

    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True
    params.randomSeed = 42

    cids = AllChem.EmbedMultipleConfs(m, numConfs=n_confs, params=params)
    if len(cids) == 0:
        raise RuntimeError("ETKDG embedding failed")

    res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=300)
    energies = [e[1] for e in res]
    k = int(np.argmin(energies))
    conf = m.GetConformer(cids[k])

    symbols, pos = [], []
    for i, a in enumerate(m.GetAtoms()):
        p = conf.GetAtomPosition(i)
        symbols.append(a.GetSymbol())
        pos.append((p.x, p.y, p.z))

    return symbols, np.array(pos, float)


def ani_optimize(symbols, positions, fmax=1e-4):
    with suppress_stdout_stderr():
        model = torchani.models.ANI2x(periodic_table_index=True)
        calc = model.ase()
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = calc
        BFGS(atoms, logfile=None).run(fmax=fmax)
    return atoms.get_chemical_symbols(), atoms.get_positions()


def ani_to_psi4_mol(symbols, positions, charge=0, multiplicity=1, symmetry="c1"):
    lines = []
    lines.append(f"{charge} {multiplicity}")
    if symmetry is not None:
        lines.append(f"symmetry {symmetry}")
    lines.append("units angstrom")
    for s, (x, y, z) in zip(symbols, positions):
        lines.append(f"{s}  {x:.8f}  {y:.8f}  {z:.8f}")
    return psi4.geometry("\n".join(lines))


def psi4_single_point(mol, n_states=5, n_threads=8, method="TD-B3LYP", basis="def2-SVP"):
    hartree_to_ev = 27.211
    
    psi4.core.clean()
    psi4.core.clean_timers()
    psi4.core.clean_options()
    psi4.set_num_threads(n_threads)
    psi4.set_options({'TDSCF_STATES': n_states, 'TDSCF_TRIPLETS': 'NONE'})
    E_s, wfn_s = psi4.energy(f'{method}/{basis}', molecule=mol, return_wfn=True)

    eps_a = np.array(wfn_s.epsilon_a())
    nalpha = wfn_s.nalpha()
    homo_E = eps_a[nalpha - 1] * hartree_to_ev
    lumo_E = eps_a[nalpha] * hartree_to_ev
    S1_E = wfn_s.variable("TD-DFT ROOT 0 -> ROOT 1 EXCITATION ENERGY") * hartree_to_ev
    S1_f = wfn_s.variable("TD-DFT ROOT 0 -> ROOT 1 OSCILLATOR STRENGTH (LEN)")

    psi4.core.clean_options()
    psi4.set_num_threads(n_threads)
    psi4.set_options({'TDSCF_STATES': n_states, 'TDSCF_TRIPLETS': 'ONLY'})
    E_t, wfn_t = psi4.energy(f'{method}/{basis}', molecule=mol, return_wfn=True)
    T1_E = wfn_t.variable("TD-DFT ROOT 0 -> ROOT 1 EXCITATION ENERGY") * hartree_to_ev

    results = {
        "HOMO": homo_E,
        "LUMO": lumo_E,
        "S1": S1_E,
        "S1_f": S1_f,
        "T1": T1_E,
        "ST_gap": (S1_E - T1_E),
    }

    order = ["HOMO", "LUMO", "S1", "S1_f", "T1", "ST_gap"]
    return np.array([float(results[k]) for k in order], dtype=float)


def predict_properties(smiles_list):
    if isinstance(smiles_list, str):
        smiles_iter = [smiles_list]
    else:
        smiles_iter = list(smiles_list)

    results = []
    for s in smiles_iter:
        charge = smiles_to_charge(s)
        multiplicity = 1

        symbols0, pos0 = smiles_to_best_xyz(s)
        symbols_opt, pos_opt = ani_optimize(symbols0, pos0)

        mol = ani_to_psi4_mol(
            symbols_opt,
            pos_opt,
            charge=charge,
            multiplicity=multiplicity
        )

        result = psi4_single_point(mol)  # shape (6,)
        results.append(result)

    return np.array(results, dtype=float)

