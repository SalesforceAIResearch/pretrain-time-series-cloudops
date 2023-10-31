import math
from typing import Tuple

import torch
import torch.nn.functional as F
from gluonts.torch.distributions import DistributionOutput
from gluonts.util import lazy_property
from torch.distributions import Distribution, constraints, Chi2


def broadcast_shape(*shapes, **kwargs):
    """
    Similar to ``np.broadcast()`` but for shapes.
    Equivalent to ``np.broadcast(*map(np.empty, shapes)).shape``.
    :param tuple shapes: shapes of tensors.
    :param bool strict: whether to use extend-but-not-resize broadcasting.
    :returns: broadcasted shape
    :rtype: tuple
    :raises: ValueError
    """
    strict = kwargs.pop("strict", False)
    reversed_shape = []
    for shape in shapes:
        for i, size in enumerate(reversed(shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
            elif reversed_shape[i] == 1 and not strict:
                reversed_shape[i] = size
            elif reversed_shape[i] != size and (size != 1 or strict):
                raise ValueError(
                    "shape mismatch: objects cannot be broadcast to a single shape: {}".format(
                        " vs ".join(map(str, shapes))
                    )
                )
    return tuple(reversed(reversed_shape))


class MultivariateStudentT(Distribution):
    arg_constraints = {
        "df": constraints.positive,
        "loc": constraints.real_vector,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, df, loc, scale_tril, validate_args=None):
        dim = loc.size(-1)
        assert scale_tril.shape[-2:] == (dim, dim)
        if not isinstance(df, torch.Tensor):
            df = loc.new_tensor(df)
        batch_shape = torch.broadcast_shapes(
            df.shape, loc.shape[:-1], scale_tril.shape[:-2]
        )
        event_shape = torch.Size((dim,))
        self.df = df.expand(batch_shape)
        self.loc = loc.expand(batch_shape + event_shape)
        self._unbroadcasted_scale_tril = scale_tril
        self._chi2 = Chi2(self.df)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self):
        # NB: this is not covariance of this distribution;
        # the actual covariance is df / (df - 2) * covariance_matrix
        return torch.matmul(
            self._unbroadcasted_scale_tril,
            self._unbroadcasted_scale_tril.transpose(-1, -2),
        ).expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self):
        identity = torch.eye(
            self.loc.size(-1), device=self.loc.device, dtype=self.loc.dtype
        )
        return torch.cholesky_solve(identity, self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @staticmethod
    def infer_shapes(df, loc, scale_tril):
        event_shape = loc[-1:]
        batch_shape = broadcast_shape(df, loc[:-1], scale_tril[:-2])
        return batch_shape, event_shape

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateStudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        scale_shape = loc_shape + self.event_shape
        new.df = self.df.expand(batch_shape)
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if "scale_tril" in self.__dict__:
            new.scale_tril = self.scale_tril.expand(scale_shape)
        if "covariance_matrix" in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(scale_shape)
        if "precision_matrix" in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(scale_shape)
        new._chi2 = self._chi2.expand(batch_shape)
        super(MultivariateStudentT, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        X = torch.empty(shape, dtype=self.df.dtype, device=self.df.device).normal_()
        Z = self._chi2.rsample(sample_shape)
        Y = X * torch.rsqrt(Z / self.df).unsqueeze(-1)
        return self.loc + self.scale_tril.matmul(Y.unsqueeze(-1)).squeeze(-1)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        n = self.loc.size(-1)
        y = torch.linalg.solve_triangular(
            self.scale_tril, (value - self.loc).unsqueeze(-1), upper=False
        ).squeeze(-1)
        Z = (
            self.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            + 0.5 * n * self.df.log()
            + 0.5 * n * math.log(math.pi)
            + torch.lgamma(0.5 * self.df)
            - torch.lgamma(0.5 * (self.df + n))
        )
        return -0.5 * (self.df + n) * torch.log1p(y.pow(2).sum(-1) / self.df) - Z

    @property
    def mean(self):
        m = self.loc.clone()
        m[self.df <= 1, :] = float("nan")
        return m

    @property
    def variance(self):
        m = self.scale_tril.pow(2).sum(-1) * (self.df / (self.df - 2)).unsqueeze(-1)
        m[(self.df <= 2) & (self.df > 1), :] = float("inf")
        m[self.df <= 1, :] = float("nan")
        return m


class MultivariateStudentTOutput(DistributionOutput):
    distr_cls = MultivariateStudentT

    def __init__(self, dims):
        super().__init__()
        self.args_dim = {
            "df": 1,
            "loc": dims,
            "scale": dims * dims,
        }

    @classmethod
    def domain_map(cls, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        df = 2.0 + F.softplus(df)
        # Lower Cholesky Transform
        d = loc.shape[-1]
        eps = torch.finfo(scale.dtype).eps
        scale = scale.view(*scale.shape[:-1], d, d).clamp_min(eps)
        scale = (
            scale.tril(-1) + F.softplus(scale.diagonal(dim1=-2, dim2=-1)).diag_embed()
        )

        return df.squeeze(-1), loc, scale

    @property
    def event_shape(self) -> Tuple:
        return (self.args_dim["loc"],)


class IndependentStudentTOutput(DistributionOutput):
    distr_cls = MultivariateStudentT

    def __init__(self, dims: int):
        super().__init__()
        self.args_dim = {
            "df": 1,
            "loc": dims,
            "scale": dims,
        }

    @classmethod
    def domain_map(cls, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        df = 2.0 + F.softplus(df)
        eps = torch.finfo(scale.dtype).eps
        scale = torch.diag_embed(F.softplus(scale).clamp_min(eps))
        return df.squeeze(-1), loc, scale

    @property
    def event_shape(self) -> Tuple:
        return (self.args_dim["loc"],)
