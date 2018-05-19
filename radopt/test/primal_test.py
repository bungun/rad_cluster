import numpy as np
import radopt as ro

def test_primal():
    TOL = 1e-10
    shape = m, n = (100, 50)
    structures = dict(
            target=ro.structure.Target('target', 99),
            oar=ro.structure.OAR('oar', 1))
    block_sizes = dict(target=m-1,  oar=1)
    dose_matrices = dict(
            target=np.random.random((m-1, n)),
            oar=np.random.random((1, n)))
    voxel_weights = dict(
            target=np.random.uniform(1, 10, m-1),
            oar=np.ones(1))

    x = np.random.random(n)
    y = {k: dose_matrices[k].dot(x) for k in dose_matrices}

    def primal_objective_consistent(**options):
        voxel_weights = options.get('voxel_weights', dict())
        norm_options = options.get('norm_options', dict())
        ff = 0
        for k, s in structures.items():
            wnorm = float(norm_options.get(k, dict()).get('weight_norm', 1))
            dnorm = float(norm_options.get(k, dict()).get('dose_norm', 1))
            vw = voxel_weights.get(k, 1)
            if isinstance(s, ro.structure.Target):
                wo = (vw/wnorm) * s.objective.parameters['weight_overdose']
                wu = (vw/wnorm) * s.objective.parameters['weight_underdose']
                dz = s.objective.parameters['dose'] / dnorm
                res = y[k] - dz
                ff += np.sum(wo * np.maximum(res, 0))
                ff += np.sum(-wu * np.minimum(res, 0))
            else:
                wt = (vw/wnorm) * s.objective.parameters['weight']
                ff += np.sum(wt * y[k])

        obj_voxels, obj_beams, f_py = ro.primal.build_objectives(
                shape, block_sizes, structures, dose_matrices, **options)
        assert abs(f_py(x) - ff) < TOL

        fpgs = 0
        offset = 0
        for k, s in structures.items():
            rng = slice(offset, offset + block_sizes[k])
            a, b, c, d, e = map(lambda v: v[rng], obj_voxels.arrays[1:-1])
            if isinstance(s, ro.structure.Target):
                fpgs += np.dot(c, np.abs(y[k] - b))
                fpgs += np.dot(d, y[k] - b)
                fpgs += np.dot(e, (y[k] - b)**2)
            else:
                fpgs += np.dot(d, y[k])
            offset += block_sizes[k]

        assert abs(f_py(x) - fpgs) < TOL
        return True

    wnorm = structures['target'].size
    opts = dict(weight_norm=wnorm, dose_norm=2.)
    nrm_opts = dict(target=opts, oar=opts)
    assert primal_objective_consistent()
    assert primal_objective_consistent(norm_options=dict(weight_norm=wnorm))
    assert primal_objective_consistent(norm_options=nrm_opts)
    assert primal_objective_consistent(
            voxel_weights=voxel_weights, norm_options=nrm_opts)
