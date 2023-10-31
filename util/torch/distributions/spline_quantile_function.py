from typing import Optional, cast

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import AffineTransform
from gluonts.core.component import validated
from gluonts.torch.distributions.distribution_output import DistributionOutput
from gluonts.torch.distributions.piecewise_linear import (
    PiecewiseLinear,
    TransformedPiecewiseLinear,
)
from gluonts.torch.distributions.isqf import ISQF, TransformedISQF


class SQFOutput(DistributionOutput):
    distr_cls: type = PiecewiseLinear

    @validated()
    def __init__(self, num_pieces: int, target_dim: int = 1) -> None:
        super().__init__(self)

        assert (
            isinstance(num_pieces, int) and num_pieces > 1
        ), "num_pieces should be an integer and greater than 1"

        self.num_pieces = num_pieces
        self.target_dim = target_dim
        self.args_dim = cast(
            dict[str, int],
            {
                "gamma": self.target_dim,
                "slopes": num_pieces * self.target_dim,
                "knot_spacings": num_pieces * self.target_dim,
            },
        )

    def domain_map(
        self,
        gamma: torch.Tensor,
        slopes: torch.Tensor,
        knot_spacings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma, slopes, knot_spacings = map(
            lambda x: rearrange(x, "... (j d) -> ... d j", d=self.target_dim).squeeze(
                -2
            ),
            (gamma, slopes, knot_spacings),
        )

        slopes_nn = torch.abs(slopes)

        knot_spacings_proj = F.softmax(knot_spacings, dim=-1)

        return gamma.squeeze(dim=-1), slopes_nn, knot_spacings_proj

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = 0,
        scale: Optional[torch.Tensor] = None,
    ) -> PiecewiseLinear:
        if scale is None:
            return self.distr_cls(*distr_args)
        else:
            distr = self.distr_cls(*distr_args)
            return TransformedPiecewiseLinear(
                distr, [AffineTransform(loc=loc, scale=scale)]
            )

    @property
    def event_shape(self) -> tuple:
        return () if self.target_dim == 1 else (self.target_dim,)


class ISQFOutput(DistributionOutput):
    r"""
    DistributionOutput class for the Incremental (Spline) Quantile Function
    Parameters
    ----------
    num_pieces
        number of spline pieces for each spline
        ISQF reduces to IQF when num_pieces = 1
    qk_x
        list containing the x-positions of quantile knots
    tol
        tolerance for numerical safeguarding
    """

    distr_cls: type = ISQF

    @validated()
    def __init__(
        self, num_pieces: int, qk_x: list[float], target_dim: int = 1, tol: float = 1e-4
    ) -> None:
        # ISQF reduces to IQF when num_pieces = 1

        super().__init__(self)

        assert (
            isinstance(num_pieces, int) and num_pieces > 0
        ), "num_pieces should be an integer and greater than 0"

        self.num_pieces = num_pieces
        self.qk_x = sorted(qk_x)
        self.num_qk = len(qk_x)
        self.target_dim = target_dim
        self.tol = tol
        self.args_dim: dict[str, int] = {
            "spline_knots": (self.num_qk - 1) * num_pieces * target_dim,
            "spline_heights": (self.num_qk - 1) * num_pieces * target_dim,
            "beta_l": 1 * target_dim,
            "beta_r": 1 * target_dim,
            "quantile_knots": self.num_qk * target_dim,
        }

    def domain_map(
        self,
        spline_knots: torch.Tensor,
        spline_heights: torch.Tensor,
        beta_l: torch.Tensor,
        beta_r: torch.Tensor,
        quantile_knots: torch.Tensor,
        tol: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Domain map function The inputs of this function are specified by
        self.args_dim.

        spline_knots, spline_heights:
        parameterizing the x-/ y-positions of the spline knots,
        shape = (*batch_shape, (num_qk-1)*num_pieces)

        beta_l, beta_r:
        parameterizing the left/right tail, shape = (*batch_shape, 1)

        quantile_knots:
        parameterizing the y-positions of the quantile knots,
        shape = (*batch_shape, num_qk)
        """

        # Add tol to prevent the y-distance of
        # two quantile knots from being too small
        #
        # Because in this case the spline knots could be squeezed together
        # and cause overflow in spline CRPS computation

        spline_knots, spline_heights, beta_l, beta_r, quantile_knots = map(
            lambda x: rearrange(x, "... (j d) -> ... d j", d=self.target_dim).squeeze(
                -2
            ),
            (spline_knots, spline_heights, beta_l, beta_r, quantile_knots),
        )

        qk_y = torch.cat(
            [
                quantile_knots[..., 0:1],
                torch.abs(quantile_knots[..., 1:]) + tol,
            ],
            dim=-1,
        )
        qk_y = torch.cumsum(qk_y, dim=-1)

        # Prevent overflow when we compute 1/beta
        beta_l = torch.abs(beta_l.squeeze(-1)) + tol
        beta_r = torch.abs(beta_r.squeeze(-1)) + tol

        return spline_knots, spline_heights, beta_l, beta_r, qk_y

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = 0,
        scale: Optional[torch.Tensor] = None,
    ) -> ISQF:
        """
        function outputing the distribution class
        distr_args: distribution arguments
        loc: shift to the data mean
        scale: scale to the data
        """

        distr_args, qk_x = self.reshape_spline_args(distr_args, self.qk_x)

        distr = self.distr_cls(*distr_args, qk_x, self.tol)

        if scale is None:
            return distr
        else:
            return TransformedISQF(distr, [AffineTransform(loc=loc, scale=scale)])

    def reshape_spline_args(self, distr_args, qk_x: list[float]):
        """
        auxiliary function reshaping knots and heights to (*batch_shape,
        num_qk-1, num_pieces) qk_x to (*batch_shape, num_qk)
        """

        spline_knots, spline_heights = distr_args[0], distr_args[1]
        batch_shape = spline_knots.shape[:-1]
        num_qk, num_pieces = self.num_qk, self.num_pieces

        # repeat qk_x from (num_qk,) to (*batch_shape, num_qk)
        qk_x_repeat = torch.tensor(
            qk_x, dtype=spline_knots.dtype, device=spline_knots.device
        ).repeat(*batch_shape, 1)

        # knots and heights have shape (*batch_shape, (num_qk-1)*num_pieces)
        # reshape them to (*batch_shape, (num_qk-1), num_pieces)
        spline_knots_reshape = spline_knots.reshape(
            *batch_shape, (num_qk - 1), num_pieces
        )
        spline_heights_reshape = spline_heights.reshape(
            *batch_shape, (num_qk - 1), num_pieces
        )

        distr_args_reshape = (
            spline_knots_reshape,
            spline_heights_reshape,
            *distr_args[2:],
        )

        return distr_args_reshape, qk_x_repeat

    @property
    def event_shape(self) -> tuple:
        return () if self.target_dim == 1 else (self.target_dim,)
