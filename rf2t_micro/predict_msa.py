"""Core predictor process."""

from typing import Optional

from collections import namedtuple
import os
import sys
import time

import numpy as np
import torch
from  torch import nn

from .parsers import _A3M_ALPHABET
from .TrunkModel  import TrunkModule

script_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
script_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))

MODEL_PARAM = {
        "n_module"     : 12,
        "n_diff_module": 12,
        "n_layer"      : 1,
        "d_msa"        : 64,
        "d_pair"       : 128,
        "d_templ"      : 128,
        "n_head_msa"   : 4,
        "n_head_pair"  : 8,
        "n_head_templ" : 4,
        "d_hidden"     : 64,
        "d_attn"       : 64,
        "d_crd"        : 64,
        "r_ff"         : 2,
        "n_resblock"   : 1,
        "p_drop"       : 0.1,
        "performer_N_opts": {},
        "performer_L_opts": {}
        }

class Predictor():

    """
    
    """
    
    def __init__(self, 
                 model_dir: Optional[str] = None, 
                 use_cpu: bool = False):
        if model_dir is None:
            self.model_dir = f"{script_dir}/weights"
        else:
            self.model_dir = model_dir
        #
        # define model name
        self.model_name = "RF2t"
        if torch.cuda.is_available() and (not use_cpu):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.active_fn = nn.Softmax(dim=1)

        # define model & load model
        self.model = TrunkModule(**MODEL_PARAM).to(self.device)
        could_load = self.load_model(self.model_name)
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()

    def load_model(self, model_name: str):
        chk_fn = f"{self.model_dir}/{model_name}.pt"
        if not os.path.exists(chk_fn):
            return False
        checkpoint = torch.load(chk_fn, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return True
    
    def predict(self, 
                msa: np.ndarray, 
                chain_a_length: int) -> np.ndarray:

        # msa = parse_a3m(paired_msa_file)
        n_rows, n_cols = msa.shape
       
        self.model.eval()
        with torch.no_grad():
            msa = torch.tensor(msa[:10_000],  # take only first 10k rows 
                               device=self.device).long().unsqueeze(0)
            idx_pdb = torch.arange(n_cols, device=self.device).long().unsqueeze(0)
            idx_pdb[:,chain_a_length:] += 200 
            seq = msa[:,0]
            #
            logit_s, _ = self.model(msa, seq, idx_pdb)
            
            # distogram
            prob = self.active_fn(logit_s[0])
            prob = prob.permute(0,2,3,1).cpu().detach().numpy()
            # interchain contact prob
        prob = np.sum(prob.reshape(n_cols,n_cols,-1)[:chain_a_length,chain_a_length:,:(len(_A3M_ALPHABET) - 1)], 
                      axis=-1)
        return prob
