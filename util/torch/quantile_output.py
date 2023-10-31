import torch
from einops import rearrange
from gluonts.torch.modules.quantile_output import QuantileOutput as _QuantileOutput
from gluonts.core.component import validated

from util.torch.ops import unsqueeze_dim


class QuantileOutput(_QuantileOutput):
    """
    Output layer using a quantile loss and projection layer to connect the
    quantile output to the network.

    Parameters
    ----------
    quantiles
        list of quantiles to compute loss over.

    quantile_weights
        weights of the quantiles.
    """

    @validated()
    def __init__(self, quantiles: list[float], target_dim: int = 1) -> None:
        super().__init__(quantiles=quantiles)
        self.target_dim = target_dim
        self.args_dim = {"quantiles": self.num_quantiles * self.target_dim}

    def domain_map(self, quantiles_pred: torch.Tensor):
        return rearrange(quantiles_pred, "... (d q) -> q ... d", d=self.target_dim)

    @property
    def event_shape(self) -> tuple[int, ...]:
        return () if self.target_dim == 1 else (self.target_dim,)

    def quantile_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean quantile loss.

        Parameters
        ----------
        y_true
            Ground truth values, shape [N_1, ..., N_k]
        y_pred
            Predicted quantiles, shape [N_1, ..., N_k num_quantiles]

        Returns
        -------
        loss
            Quantile loss, shape [N_1, ..., N_k]
        """
        y_true = y_true.unsqueeze(0)
        quantiles = unsqueeze_dim(torch.tensor(
            self.quantiles, dtype=y_pred.dtype, device=y_pred.device
        ), y_pred.shape[1:])
        return 2 * (
            (y_true - y_pred) * ((y_true <= y_pred).float() - quantiles)
        ).abs().sum(dim=0)
