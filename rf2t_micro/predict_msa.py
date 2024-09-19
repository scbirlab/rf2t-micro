"""Core predictor process."""

from typing import Optional

from collections import namedtuple
import os
import sys
import time

import numpy as np
from numpy.typing import ArrayLike
import torch
from  torch import nn

from .parsers import _A3M_ALPHABET
from .TrunkModel  import TrunkModule
from .weights import get_model_weights

MODEL_PARAM = {
        "n_module"        : 12,
        "n_diff_module"   : 12,
        "n_layer"         : 1,
        "d_msa"           : 64,
        "d_pair"          : 128,
        "d_templ"         : 128,
        "n_head_msa"      : 4,
        "n_head_pair"     : 8,
        "n_head_templ"    : 4,
        "d_hidden"        : 64,
        "d_attn"          : 64,
        "d_crd"           : 64,
        "r_ff"            : 2,
        "n_resblock"      : 1,
        "p_drop"          : 0.1,
        "performer_N_opts": {},
        "performer_L_opts": {}
    }

class Predictor:

    """Class to contain model weights and make predictions on supplied paired MSAs.

    Parameters
    ----------
    use_cpu : bool, optional
        Whether to force CPU usage. Default: use GPU if available.
    return_logits : bool, optional
        Whether to return values from `(–inf,inf)` or sigmoid-transformed to `(–1, 1)`.
        Default: `False`, return sigmoid-transformed.
    model_dir : str, Optional
        Directory containing model weights in filename `"RF2t.pt"`.
    
    Returns
    -------
    Predictor
        An object which can make predictions on paired MSA objects.
    
    """
    
    def __init__(self, 
                 model_dir: Optional[str] = None, 
                 use_cpu: bool = False,
                 return_logits: bool = False):
        self.return_logits = return_logits
        if model_dir is None:
            self.model_dir = get_model_weights()
        else:
            self.model_dir = get_model_weights(model_dir)
        #
        # define model name
        self.model_name = "RF2t"
        if torch.cuda.is_available() and (not use_cpu):
            self.device_type = 'cuda'
        else:
            self.device_type = 'cpu'
        self.device = torch.device(self.device_type)
        self.active_fn = nn.Softmax(dim=1)

        # define model & load model
        self.model = TrunkModule(**MODEL_PARAM).to(self.device)
        could_load = self.load_model(self.model_name)
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()

    def load_model(self, model_name: str) -> bool:
        weights_filename = os.path.join(self.model_dir, f"{model_name}.pt")
        if not os.path.exists(weights_filename):
            return False
        checkpoint = torch.load(weights_filename, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return True

    def to(self, device_type: str, *args):
        self.device_type = device_type
        self.device = torch.device(self.device_type)
        self.model.to(self.device)
        for arg in args:
            arg.to(self.device)
        return None
    
    def predict(self, 
                msa: ArrayLike, 
                chain_a_length: int,
                max_msa_depth: int = 10_000) -> np.ndarray:

        n_rows, n_cols = msa.shape
        self.model.eval()
        try:
            with torch.no_grad():
                msa_tensor = torch.tensor(msa[:max_msa_depth],  # take only first 10k rows 
                                          device=self.device).long().unsqueeze(0)
                idx_pdb = torch.arange(n_cols, device=self.device).long().unsqueeze(0)
                idx_pdb[:,chain_a_length:] += 200
                try:
                    seq = msa_tensor[:,0]  # first column?
                except IndexError as e:
                    print(msa)
                    print(msa_tensor)
                    print(max_msa_depth)
                    raise e

                # with torch.autocast(device_type=self.device_type):
                logit_s, cα_coords = self.model(msa_tensor, seq, idx_pdb)
                
                # distogram
                if not self.return_logits:
                    prob = self.active_fn(logit_s[0])
                else:
                    prob = logit_s[0]
                prob = prob.permute(0, 2, 3, 1)
                # residue contact prob
                prob = prob.reshape(n_cols, n_cols, -1)[...,:(len(_A3M_ALPHABET) - 1)]
                prob = prob.sum(dim=-1).detach().cpu().numpy()
        except torch.OutOfMemoryError:
            self.to('cpu')
            return self.predict(msa, chain_a_length, max_msa_depth=max_msa_depth)
        else:
            return prob, cα_coords
        
