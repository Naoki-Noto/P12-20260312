# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 16:20:16 2025

@author: noton
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rdkit import Chem
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained("T5_storage/Pretraining_1M_4epochs/Frag2OPST5_1M/tokenizer")
Frag2OPST5 = AutoModelForSeq2SeqLM.from_pretrained("T5_storage/Pretraining_1M_4epochs/Frag2OPST5_1M/model")
Frag2OPST5.eval()
Frag2OPST5.config.use_cache = True
Frag2OPST5.to("cuda")


def generator(frag_list, max_new_tokens=128):
    combo = ".".join(frag_list)
    device = next(Frag2OPST5.parameters()).device
    enc = tokenizer([combo], return_tensors="pt").to(device)

    Frag2OPST5.eval()
    with torch.inference_mode():
        ids = Frag2OPST5.generate(**enc, do_sample=False, num_beams=1, num_return_sequences=1,
                             max_new_tokens=max_new_tokens, early_stopping=False, use_cache=True)
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def gen_smiles_from_fragments(frag_list, max_new_tokens=128):
    s = generator(frag_list, max_new_tokens=max_new_tokens)
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
