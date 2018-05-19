r""" Methods to build treatment objectives parameterized by weights and doses

    We assume an radiation treatment intensity optimization problem of
    the form:

     (1) minimize  :math:`\sum_{s \in S} f_s(y_s)`
              s.t. :math:`y_s = A_s x    \forall s \in S`
                   :math:`x \ge 0`,

    where S is a set of structures (problem blocks), f_s is a
    structure-specific treatment objective, and A_s is a
    structure-specifc dose matrix. In addition to these problem data,
    we have the optimization variables x (beam intensities) and y_s
    (voxel doses).

    This problem has the dual,

     (2)   maximize :math:`\sum_{s \in S} -f^*_s(\nu_s)`
               s.t. :math:`\mu = \sum_{s \in S} A_s^T \nu_s`
                    :math:`\mu \ge 0`
                    :math:`\nu_s \dom(f^*_s)`

    Where f^*_s is the convex conjugate of f_s, and the optimization
    variables are mu (beam prices) and nu_s (voxel prices). Note that
    we assume clients of this module will minimize the negative of the
    dual objective, so we simply provide f^*_s.

    Each concrete instance of Objective provides
    - a method to build a POGSObjective corresponding to :math:`f_s`
    - a method to build a POGSObjective corresponding to :math:`f^*_s`
    fused with the indicator :math:`I_{z in dom(f^*)}`
    - methods to build numpy-backed functions evaluating the above

    We further admit a perturbed dual problem

     (3)   maximize :math:`\sum_{s \in S} -f^*_s(\delta + \nu_{offset})`
               s.t. :math:`\mu = \sum_{s \in S} A_s^T (delta + \nu_{offset})`
                    :math:`\mu \ge 0`
                    :math:`(\delta + \nu_{offset} \dom(f^*_s)`

    where the optimization variable is :math:`\delta` and
    :math:`\nu_offset` is problem data. When ``nu_offset`` is provided
    as a keyword to the dual evaluation/constraint function factories,
    we return dual functions and constraints that incorporate the offset
    and take :math:`delta` as an input.

    One option final option for the perturbed dual problem is to add
    the constraint $\delta >= 0$, in which case the dual feasibility
    evaluation factory builds the indicator :math:`I_F`, where the set
    F is defined as

        :math:`F = dom(f^*) \cup \{z \mid z \ge 0\}`
"""
import collections
import abc
import six
import numpy as np
import optkit as ok

DOSE_PWL_DEFAULT = 1.0
WT_PWL_UNDER_DEFAULT = 1.0
WT_PWL_OVER_DEFAULT = 0.05
WT_OAR_DEFAULT = 0.03

@six.add_metaclass(abc.ABCMeta)
class Objective(object):
    def __init__(self, *keys_and_defaults, **parameters):
        self.parameters = dict()
        for key, default in keys_and_defaults:
            p = parameters.pop(key, default)
            self.parameters[key] = max(0, float(p))
        self._primal_param_calcs = list()
        self._dual_param_calcs = list()

    def normalized_params(self, weight_norm=1, dose_norm=1, **options):
        weight_norm = float(weight_norm)
        assert weight_norm > 0, "Weight normalization strictly positive"

        dose_norm = float(dose_norm)
        assert dose_norm > 0, "Dose normalization strictly positive"

        norm_params = dict()
        for k in self.parameters:
            if str(k).startswith('weight'):
                norm_params[k] = self.parameters[k] / weight_norm
            elif str(k).startswith('dose'):
                norm_params[k] = self.parameters[k] / dose_norm
        return norm_params

    @abc.abstractmethod
    def mix_voxel_weights(self, **params):
        raise NotImplementedError

    def calculate_primal_params(self, size, **options):
        voxel_weights = options.pop('voxel_weights', 1.)
        params = self.normalized_params(**options)
        params['voxel_weights'] = voxel_weights
        params['size'] = size
        params = self.mix_voxel_weights(**params)
        for calc in self._primal_param_calcs:
            params.update(calc(**params))
        return params

    def calculate_dual_params(self, size, **options):
        params = self.calculate_primal_params(size, **options)
        params['nu_offset'] = options.pop('nu_offset', 0.)
        params['nonnegative'] = options.pop('nonnegative', False)
        for calc in self._dual_param_calcs:
            params.update(calc(**params))
        return params

    @abc.abstractmethod
    def pogs_primal(self, **params):
        raise NotImplementedError

    @abc.abstractmethod
    def py_primal(self, **params):
        raise NotImplementedError

    @abc.abstractmethod
    def pogs_dual(self, **params):
        raise NotImplementedError

    @abc.abstractmethod
    def py_dual(self, **params):
        raise NotImplementedError

    def build_primal(self, size, **options):
        r""" Construct primal objective :math:`f(z)`

            returns ok.api.PogsObjective corresponding to this expression,
            as well as a python function that maps z->f(z)
        """
        params = self.calculate_primal_params(size, **options)
        return self.pogs_primal(**params), self.py_primal(**params)

    def build_dual(self, size, **options):
        """ Construct dual objective f_{conj}(\nu) and dual domain constraints

            returns ok.api.PogsObjective corresponding to this expression
            including the constraint, as well two python functions,
            (1) one that maps \nu->f_{conj}(\nu) and
            (2) another that implements the indicator \nu \in dom(f_{conj})

            if ``nu_offset`` supplied as a keyword argument, we build
            representations of the modified objective and constraint:

                (1) f^*(\nu + \nu_offset)
                (2) nu + nu_offset \in dom(f^*)

            if ``nonnegative=True`` specified as a keyword argument,
            the constraint \nu >= 0 will be added to the representations
        """
        params = self.calculate_dual_params(size, **options)
        pogs_d = self.pogs_dual(**params)
        py_d, py_feas = self.py_dual(**params)
        return pogs_d, py_d, py_feas

    @abc.abstractmethod
    def _fn_string(self):
        raise NotImplementedError

    def __repr__(self):
        return type(self).__name__

    def __str__(self):
        name = OBJECTIVE_NAMES[repr(self)]
        return '{}\n\tform: f(z) = {}\n\tparameters: {}'.format(
                repr(self), self._fn_string(), self.parameters)

class ObjectiveLinear(Objective):
    def __init__(self, **params):
        Objective.__init__(self, ('weight', WT_OAR_DEFAULT), **params)
        self._primal_param_calcs.append(self.linear)

    def mix_voxel_weights(self, **params):
        params['weight'] = params['weight'] * params['voxel_weights']
        return params

    def linear(self, collapse=True, **params):
        if collapse:
            params['size'] = 1
        # else:
            # raise ValueError('linear objective should be collapsed')
        return params

    def pogs_primal(self, **params):
        return ok.api.PogsObjective(
                params['size'], h='Zero', c=0, d=params['weight'])

    def py_primal(self, **params):
        r""" :math:`f(z) = w^Tz` """
        def primal_eval(y):
            return float(np.sum(params['weight'] * y))
        return primal_eval

    def pogs_dual(self, **params):
        wt = params['weight'] - params['nu_offset']
        return ok.api.PogsObjective(params['size'], h='IndEq0', b=wt)

    def py_dual(self, **params):
        r""" :math:`f^*(\tilde z) = 0`

            domain constraint: :math:`z \in {\nu: \nu == w}`
        """
        dual_eval = lambda nu: 0.
        def dual_feas(nu, tol):
            nu0 = params['nu_offset']
            wt = params['weight']
            return np.all(np.abs(nu + nu0 - wt) < tol)
        return dual_eval, dual_feas

    def _fn_string(self):
        return "w'z"

class ObjectivePWL(Objective):
    def __init__(self, **params):
        Objective.__init__(
                self,
                ('weight_underdose', WT_PWL_UNDER_DEFAULT),
                ('weight_overdose', WT_PWL_OVER_DEFAULT),
                ('dose', DOSE_PWL_DEFAULT),
                **params)
        self._primal_param_calcs.append(self.pwl_to_abs)
        self._dual_param_calcs.append(self.dual_box)

    def mix_voxel_weights(self, **params):
        vw = params['voxel_weights']
        params['weight_underdose'] = params['weight_underdose'] * vw
        params['weight_overdose'] = params['weight_overdose'] * vw
        return params

    def pwl_to_abs(self, **params):
        wt_under = params['weight_underdose']
        wt_over = params['weight_overdose']
        params['weight_abs'] = (wt_over + wt_under) / 2
        params['weight_lin'] = (wt_over - wt_under) / 2
        return params

    def pogs_primal(self, **params):
        return ok.api.PogsObjective(
                params['size'], h='Abs',
                b=params['dose'], c=params['weight_abs'],
                d=params['weight_lin'])

    def py_primal(self, **params):
        r""" :math:`f(z) = w_+^T(z - d)_+  -w_-^T(z - d)_-`

            in particular, this equivalent to

                :math:`f_(z) = wt_{abs}|z-d| + w_{lin}(z-d)`

            for

                :math:`w_+ = w_{abs} + w_{lin}`
                :math:`w_- = w_{abs} - w_{lin}`.
        """
        def primal_eval(y):
            wt_abs = params['weight_abs']
            wt_lin = params['weight_lin']
            res = (y - params['dose'])
            return np.sum(wt_abs * np.abs(res)) + np.sum(wt_lin * res)
        return primal_eval

    def dual_box(self, nonnegative=False, nu_offset=0, **params):
        r""" :math:`|\nu + \nu_{offset} - wt_lin|_\infty <= wt_abs`"""

        # w_lin - w_abs <= nu + nu0 <= w_lin + w_abs
        # -w_under <= nu + nu0 <= w_over
        lim_upper = params['weight_overdose'] - nu_offset
        lim_lower = -params['weight_underdose'] - nu_offset
        if nonnegative:
            lim_lower = np.maximum(lim_lower, 0)
        # width = UL - LL
        width = np.round(lim_upper - lim_lower, decimals=4)
        # scaling = 1/width
        scaling = 0. * width
        try:
            for i, wd in enumerate(width):
                scaling[i] = 1./wd if wd > 0 else 0.
        except:
            scaling = 1./width if width > 0 else 0.
        scaling = np.round(scaling, decimals=8)
        return dict(
                lim_lower=lim_lower, lim_upper=lim_upper, scaling=scaling,
                offset=scaling * lim_lower)

    def pogs_dual(self, **params):
        size = params['size']
        scaling = params['scaling']
        offset = params['offset']
        dose = params['dose']

        try:
            h = ['IndBox01'] * size
            for i, si in enumerate(scaling):
                if si == 0:
                    h[i] = 'IndEq0'
                    scaling[i] = 1.
            return ok.api.PogsObjective(
                    size, h=h, a=scaling, b=offset, d=dose)
        except:
            return ok.api.PogsObjective(
                    size, h='IndBox01', a=scaling, b=offset, d=dose)


    def py_dual(self, nu_offset=0, **params):
        """ :math:`f_{conj}(\nu) = d^T\nu`, and :math:`|\nu - wt_lin|_\infty <= wt_abs`
        """
        dose_targ = params['dose']
        def dual_eval(nu):
            return np.sum(dose_targ * (nu + nu_offset))

        def dual_feas(nu, tol):
            lim_lower = params['lim_lower']
            lim_upper = params['lim_upper']
            return np.all(nu > lim_lower - tol) and np.all(nu < lim_upper + tol)
        return dual_eval, dual_feas

    def _fn_string(self):
        return "1/2 (w_under + w_over)'|z-d| + 1/2 (w_under - w_over)'(z-d)"

OBJECTIVE_CONSTRUCTORS = dict(
        piecewise_linear=ObjectivePWL,
        linear=ObjectiveLinear,
)

OBJECTIVE_NAMES = dict(
        ObjectivePWL='piecewise_linear',
        ObjectiveLinear='linear',
)