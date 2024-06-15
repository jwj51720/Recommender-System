import torch
import torch.nn as nn

# fmt:off
class FactorizationMachine(nn.Module):
    def __init__(self, num_features: int, num_factors: int) -> None:
        """
        FactorizationMachine 초기화

        Args:
            num_features (int): dims of Input
            num_factors (int): dimens of Factorization
        """
        super(FactorizationMachine, self).__init__()
        self.n = num_features
        self.k = num_factors

        self.w0 = nn.Parameter(torch.zeros(1))  # global bias
        self.w = nn.Parameter(torch.zeros(self.n))  # strength of the i-th variable
        self.V = nn.Parameter(torch.randn(self.n, self.k))  # i-th variable with k factors

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.w)
        nn.init.xavier_uniform_(self.V)

    def forward(self, x):
        assert len(x.shape) == 2 # [B, n]
        linear = self.w0 + torch.matmul(x, self.w)  # [B]
        vx_2 = torch.pow(torch.matmul(x, self.V), 2) # [B, k]
        v_2x_2 = torch.matmul(x ** 2, self.V ** 2) # [B, k]
        interaction = torch.sum(vx_2 - v_2x_2, dim = 1) / 2 # [B]
        y = linear + interaction # [B]
        return y
# fmt:on
