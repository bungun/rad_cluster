import numpy as np
import radopt as ro

def test_objective_base():
    class ObjectiveMock(ro.objective.Objective):
        def _fn_string(self, *args, **kwargs): return 'mock'
        def mix_voxel_weights(self, **kwargs):
            return {k: ('mock', kwargs[k]) for k in kwargs}
        def pogs_dual(self, *args, **kwargs): return 'pogs_dual'
        def pogs_primal(self, *args, **kwargs): return 'pogs_primal'
        def py_dual(self, *args, **kwargs): return 'py_dual', 'py_feas'
        def py_primal(self, *args, **kwargs): return 'py_primal'

    obj = ObjectiveMock()
    assert obj.build_dual(10) == ('pogs_dual', 'py_dual', 'py_feas')
    assert obj.build_primal(10) == ('pogs_primal', 'py_primal')
    expect = dict(nonnegative=False, nu_offset=0)
    expect['voxel_weights'] = ('mock', 1)
    expect['size'] = ('mock', 10)
    assert obj.calculate_dual_params(10) == expect
    expect.pop('nonnegative')
    expect.pop('nu_offset')
    assert obj.calculate_primal_params(10) == expect
    assert obj.normalized_params(1., 1., **dict()) == dict()
    assert obj.parameters == dict()

def test_objective_linear():
    obj = ro.objective.ObjectiveLinear()
    assert obj.parameters == dict(weight=ro.objective.WT_OAR_DEFAULT)

    obj2 = ro.objective.ObjectiveLinear(weight=10)
    assert obj2.parameters == dict(weight=10)

    def exercise_objective(size, weight, **options):
        obj_pogs, obj_py = obj.build_primal(size, **options)
        x = np.random.random(size)
        fx = obj_py(x)
        expect = weight * np.sum(x)
        assert fx - expect < 1e-10
        assert fx - np.sum(obj_pogs.d * x) < 1e-10

        obj_pogs_d, obj_py_d, feas_py_d = obj.build_dual(size, **options)
        weight -= options.pop('nu_offset', 0)
        assert not feas_py_d(x, 1e-3)
        assert feas_py_d(weight, 1e-3)
        assert obj_py_d(x) == 0
        assert np.sum(obj_pogs_d.a - 1) < 1e-10
        assert np.sum(obj_pogs_d.b - weight) < 1e-10
        assert np.sum(obj_pogs_d.c - 1) < 1e-10
        assert np.sum(obj_pogs_d.d) < 1e-10
        assert np.sum(obj_pogs_d.e) < 1e-10
        return True

    wt = obj.parameters['weight']
    assert exercise_objective(10, wt)
    assert exercise_objective(10, 1, weight_norm=wt)
    assert exercise_objective(10, 0.5, weight_norm=2*wt)
    d_rand = np.random.random()
    assert exercise_objective(10, wt, dose_norm=d_rand)
    assert exercise_objective(10, 1, weight_norm=wt, dose_norm=d_rand)
    nu_rand_scal = np.random.random()
    nu_rand = nu_offset=np.random.random((1,1))
    assert exercise_objective(10, 1, weight_norm=wt, nu_offset=nu_rand_scal)
    assert exercise_objective(10, 1, weight_norm=wt, nu_offset=nu_rand)
    assert exercise_objective(
            10, 1, weight_norm=wt, nu_offset=nu_rand_scal, nonnegative=True)

def test_objective_pwl():
    obj = ro.objective.ObjectivePWL()
    assert obj.parameters == dict(
            weight_overdose=ro.objective.WT_PWL_OVER_DEFAULT,
            weight_underdose=ro.objective.WT_PWL_UNDER_DEFAULT,
            dose=ro.objective.DOSE_PWL_DEFAULT,)

    obj2 = ro.objective.ObjectivePWL(
            weight_underdose=10, weight_overdose=5, dose=3)
    assert obj2.parameters == dict(
            weight_underdose=10, weight_overdose=5, dose=3)

    def exercise_objective(size, weight_under, weight_over, dose, **options):
        TOL = options.pop('tol', 1e-6)
        obj_pogs, obj_py = obj.build_primal(size, **options)
        x = np.random.normal(dose, 0.5, size)
        res = x - dose
        res_pos = np.maximum(res, 0)
        res_neg = np.minimum(res, 0)
        fx = obj_py(x)
        expect = np.sum(weight_under * -res_neg) + np.sum(weight_over * res_pos)
        pgs = np.dot(obj_pogs.c, np.abs(x - dose)) + np.dot(obj_pogs.d, x - dose)
        assert abs(fx - expect) < TOL * abs(fx), '{}, {}'.format(fx, expect)
        assert abs(fx - pgs) < TOL

        obj_pogs_d, obj_py_d, feas_py_d = obj.build_dual(size, **options)

        # |nu + nu_offset - wt_lin| < wt_abs
        offset = options.pop('nu_offset', 0)
        nonnegative = options.pop('nonnegative', False)

        lower_lim = obj_pogs.d - obj_pogs.c - offset
        upper_lim = obj_pogs.d + obj_pogs.c - offset
        if nonnegative:
            lower_lim = np.maximum(lower_lim, 0)
        width = np.round(upper_lim - lower_lim, decimals=4)

        alpha = np.random.uniform(0, 1)
        nu_feasible = alpha * lower_lim + (1-alpha) * upper_lim

        strlim = '\n'
        for i in range(len(nu_feasible)):
            strlim += '{} < {} < {}\n'.format(lower_lim[i], nu_feasible[i], upper_lim[i])
        assert feas_py_d(nu_feasible, TOL), strlim

        def scal_equal(first, second):
            return abs(first - second) <= TOL * abs(first)
        def vec_equal(first, second):
            return np.linalg.norm(first - second) <= TOL * np.linalg.norm(first)

        dual_obj = -obj_py_d(nu_feasible)
        obj_expect = -np.sum(dose * (nu_feasible + offset))
        a_expect = 1./width
        b_expect = lower_lim/width
        assert scal_equal(dual_obj, obj_expect), '{} == {}'.format(dual_obj, obj_expect)
        assert vec_equal(obj_pogs_d.a, a_expect), '{} == {}'.format(obj_pogs_d.a, a_expect)
        assert vec_equal(obj_pogs_d.b, b_expect), '{} == {}'.format(obj_pogs_d.b, b_expect)
        assert vec_equal(obj_pogs_d.c, 1), '{} == {}'.format(obj_pogs_d.c, 1)
        assert vec_equal(obj_pogs_d.d, dose), '{} == {}'.format(obj_pogs_d.d, dose)
        assert vec_equal(obj_pogs_d.e, 0.), '{} == {}'.format(obj_pogs_d.e, 0)
        return True

    wtu = obj.parameters['weight_underdose']
    wto = obj.parameters['weight_overdose']
    dz = obj.parameters['dose']
    assert exercise_objective(10, wtu, wto, dz)
    assert exercise_objective(10, 1, wto/wtu, dz, weight_norm=wtu)
    assert exercise_objective(10, 0.5, wto/(2*wtu), dz, weight_norm=2*wtu)
    dnorm = np.random.random()
    assert exercise_objective(10, wtu, wto, dz/dnorm, dose_norm=dnorm)
    assert exercise_objective(10, 1, wto/wtu, dz/dnorm, weight_norm=wtu, dose_norm=dnorm)
    nu_offset = np.random.uniform(0, wto, 10) # some feasible -w_lower < nu < w_upper
    assert exercise_objective(10, 1, wto/wtu, dz, weight_norm=wtu, nu_offset=nu_offset)
    assert exercise_objective(10, 1, wto/wtu, dz, weight_norm=wtu, nu_offset=nu_offset, nonnegative=True)

    vw = np.random.random(10)
    assert exercise_objective(10, wtu * vw, wto * vw, dz, voxel_weights=vw)
    assert exercise_objective(10, vw, vw * wto/wtu, dz, voxel_weights=vw, weight_norm=wtu)

