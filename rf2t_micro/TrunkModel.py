"""Main model definition."""

from typing import Optional

import torch
from torch.nn import Module

from .Embeddings import MSA_emb, Pair_emb_wo_templ, Pair_emb_w_templ, Templ_emb
from .Attention_module import IterativeFeatureExtractor
from .DistancePredictor import DistanceNetwork
from .InitStrGenerator import InitStr_Network

class TrunkModule(Module):
    """Main model definition.
    
    """

    def __init__(self, 
                 n_module: int = 4, 
                 n_diff_module: int = 2, 
                 n_layer: int = 4,
                 d_msa: int = 64, 
                 d_pair: int = 128, 
                 d_templ: int = 4,
                 n_head_msa: int = 4, 
                 n_head_pair: int = 8, 
                 n_head_templ: int = 4,
                 d_hidden: int = 64, 
                 d_attn: int = 50, 
                 d_crd: int = 64,
                 r_ff: int = 4, 
                 n_resblock: int = 1, 
                 p_drop: float = 0.1, 
                 performer_L_opts: Optional = None, 
                 performer_N_opts: Optional = None):
        super().__init__()
        self.msa_emb = MSA_emb(d_model=d_msa, 
                               p_drop=p_drop, 
                               max_len=5000)
        self.pair_emb = Pair_emb_wo_templ(d_model=d_pair)
        self.feat_extractor = IterativeFeatureExtractor(
            n_module=n_module,
            n_diff_module=n_diff_module,
            n_layer=n_layer,
            d_msa=d_msa, d_pair=d_pair,
            n_head_msa=n_head_msa,
            n_head_pair=n_head_pair,
            r_ff=r_ff,
            n_resblock=n_resblock,
            p_drop=p_drop,
            performer_N_opts=performer_N_opts,
            performer_L_opts=performer_L_opts,
        )
        self.c6d_predictor = DistanceNetwork(1, d_pair, 
                                             block_type='bottle', 
                                             p_drop=p_drop)
        # TODO
        self.crd_predictor = InitStr_Network(
            d_model=d_pair, 
            d_hidden=d_hidden, 
            d_attn=d_attn, 
            d_out=d_crd, 
            d_msa=d_msa,
            n_layers=n_layer, 
            n_att_head=n_head_msa, 
            p_drop=p_drop,
            performer_opts=performer_L_opts,
        )

    def forward(self, 
                msa, seq, idx):
        batch_size, n_row, n_col = msa.shape
        # Get embeddings
        msa = self.msa_emb(msa, idx)
        pair = self.pair_emb(seq, idx)
        # Extract features
        msa, pair = self.feat_extractor(msa, pair)
        # Predict 3D coordinates of CA atoms # TODO
        crds = self.crd_predictor(msa, pair, seq, idx)

        # Predict 6D coords
        pair = pair.view(batch_size, n_col, n_col, -1).permute(0, 3, 1, 2) # (batch_size, C, n_col, n_col) 
        logits = self.c6d_predictor(pair.contiguous())
        #return logits
        return logits, crds.view(batch_size, n_col, 3, 3) # TODO
