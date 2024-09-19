
from torch.nn import Module

from .resnet import ResidualNetwork

# predict distance map from pair features
# based on simple 2D ResNet

class DistanceNetwork(Module):
    def __init__(
        self, 
        n_block, 
        n_feat, 
        block_type: str = 'orig', 
        p_drop: float = .1
    ):
        super().__init__()
        self.resnet_dist = ResidualNetwork(n_block, n_feat, n_feat, 37, block_type=block_type, p_drop=p_drop)
        self.resnet_omega = ResidualNetwork(n_block, n_feat, n_feat, 37, block_type=block_type, p_drop=p_drop)
        self.resnet_theta = ResidualNetwork(n_block, n_feat, n_feat, 37, block_type=block_type, p_drop=p_drop)
        self.resnet_phi = ResidualNetwork(n_block, n_feat, n_feat, 19, block_type=block_type, p_drop=p_drop)

    def forward(self, x):
        # input: pair info (1, C, L, L)

        # predict theta, phi (non-symmetric)
        logits_theta = self.resnet_theta(x)
        logits_phi = self.resnet_phi(x)

        # predict dist, omega
        x = 0.5 * (x + x.permute(0,1,3,2))
        logits_dist = self.resnet_dist(x)
        logits_omega = self.resnet_omega(x)

        return logits_dist, logits_omega, logits_theta, logits_phi
