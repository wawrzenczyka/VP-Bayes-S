import chainer.functions as F
import numpy as np
from chainer import Variable, function
from chainer.backends import cuda
from chainer.utils import type_check


class PULoss(function.Function):
    """wrapper of loss function for PU learning"""

    def __init__(
        self,
        prior,
        loss=(lambda x: F.sigmoid(-x)),
        gamma=1,
        beta=0,
        nnpu=True,
        single_sample=False,
    ):
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss
        self.nnpu = nnpu
        self.single_sample = single_sample

        self.x_in = None
        self.x_out = None
        self.loss = None
        self.positive = 1
        self.unlabeled = -1

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype == np.float32,
            t_type.dtype == np.int32,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0],
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        t = t[:, None]
        positive, unlabeled = t == self.positive, t == self.unlabeled
        n_positive, n_unlabeled = max([1.0, xp.sum(positive)]), max(
            [1.0, xp.sum(unlabeled)]
        )
        n = n_positive + n_unlabeled

        self.x_in = Variable(x)
        y_positive = self.loss_func(self.x_in)
        y_unlabeled = self.loss_func(-self.x_in)

        if not self.single_sample:
            positive_risk = self.prior * F.sum(positive * y_positive) / n_positive
            negative_risk = (
                F.sum((unlabeled * y_unlabeled)) / n_unlabeled
                - self.prior * F.sum(positive * y_unlabeled) / n_positive
            )
        else:
            positive_risk = self.prior * 1 / n_positive * F.sum(positive * y_positive)
            negative_risk = (
                (n_unlabeled / n) * (1 / n_unlabeled) * F.sum(unlabeled * y_unlabeled)
            ) - (
                np.maximum(0, self.prior - n_positive / n)
                * (1 / n_positive)
                * F.sum(positive * y_unlabeled)
            )
        objective = positive_risk + negative_risk
        if self.nnpu:
            if negative_risk.data < -self.beta:
                objective = positive_risk - self.beta
                self.x_out = -self.gamma * negative_risk
            else:
                self.x_out = objective
        else:
            self.x_out = objective
        self.loss = xp.array(objective.data, dtype=self.x_out.data.dtype)
        return (self.loss,)

    def backward(self, inputs, gy):
        self.x_out.backward()
        gx = (
            gy[0].reshape(gy[0].shape + (1,) * (self.x_in.data.ndim - 1))
            * self.x_in.grad
        )
        return gx, None


def pu_loss(
    x, t, prior, loss=(lambda x: F.sigmoid(-x)), nnpu=True, single_sample=False
):
    """wrapper of loss function for non-negative/unbiased PU learning

        .. math::
            \\begin{array}{lc}
            L_[\\pi E_1[l(f(x))]+\\max(E_X[l(-f(x))]-\\pi E_1[l(-f(x))], \\beta) & {\\rm if nnPU learning}\\\\
            L_[\\pi E_1[l(f(x))]+E_X[l(-f(x))]-\\pi E_1[l(-f(x))] & {\\rm otherwise}
            \\end{array}

    Args:
        x (~chainer.Variable): Input variable.
            The shape of ``x`` should be (:math:`N`, 1).
        t (~chainer.Variable): Target variable for regression.
            The shape of ``t`` should be (:math:`N`, ).
        prior (float): Constant variable for class prior.
        loss (~chainer.function): loss function.
            The loss function should be non-increasing.
        nnpu (bool): Whether use non-negative PU learning or unbiased PU learning.
            In default setting, non-negative PU learning will be used.

    Returns:
        ~chainer.Variable: A variable object holding a scalar array of the
            PU loss.

    See:
        Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi Sugiyama.
        "Positive-Unlabeled Learning with Non-Negative Risk Estimator."
        Advances in neural information processing systems. 2017.
        du Plessis, Marthinus Christoffel, Gang Niu, and Masashi Sugiyama.
        "Convex formulation for learning from positive and unlabeled data."
        Proceedings of The 32nd International Conference on Machine Learning. 2015.
    """
    return PULoss(prior=prior, loss=loss, nnpu=nnpu, single_sample=single_sample)(x, t)
