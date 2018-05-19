import numpy as np
import radopt as ro

def test_dual():
    TOL = 1e-10
    FEAS_TOL = 1e-3
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

    nu = {k: np.random.normal(0, 1, (structures[k].size, )) for k in structures}
    nu_offset = {k: np.random.random() * nu[k] for k in structures}
    nu_vec = np.hstack((nu[k] for k in structures))
    mu = sum((dose_matrices[k].T.dot(nu[k]) for k in structures))

    def dual_objective_consistent(**options):
        nu_offset = options.get('nu_offset', dict())
        voxel_weights = options.get('voxel_weights', dict())
        norm_options = options.get('norm_options', dict())
        ff = 0
        ffeas = True
        for k, s in structures.items():
            wnorm = float(norm_options.get(k, dict()).get('weight_norm', 1))
            dnorm = float(norm_options.get(k, dict()).get('dose_norm', 1))
            nu0 = nu_offset.get(k, 0)
            nuuk = nu[k] + nu0
            nonnegative = k in nu_offset
            vw = voxel_weights.get(k, 1)
            if isinstance(s, ro.structure.Target):
                wo = (vw/wnorm) * s.objective.parameters['weight_overdose']
                wu = (vw/wnorm) * s.objective.parameters['weight_underdose']
                dz = s.objective.parameters['dose'] / dnorm
                ff += np.sum(dz * nuuk)
                lower_lim = -wu
                if nonnegative:
                    lower_lim = np.maximum(lower_lim, 0)
                upper_lim = wo
                ffeas &= np.all(nuuk > lower_lim - FEAS_TOL)
                ffeas &= np.all(nuuk < upper_lim + FEAS_TOL)
            else:
                wt = (vw/wnorm) * s.objective.parameters['weight']
                ff += 0
                ffeas &= np.all(abs(nuuk - wt) < FEAS_TOL)

        obj_voxels, obj_beams, fstar_py, feas_py = ro.dual.build_objectives(
                shape, block_sizes, structures, dose_matrices, **options)

        assert abs(fstar_py(nu_vec) - fstar_py(nu)) < TOL
        assert abs(fstar_py(nu_vec) - ff) < TOL
        assert feas_py(nu_vec, FEAS_TOL, verbose=False) == feas_py(nu, FEAS_TOL, verbose=False)
        feas_domain, feas_cone = feas_py(nu_vec, FEAS_TOL, verbose=False)
        assert feas_domain == ffeas
        assert feas_cone == np.all(mu > -FEAS_TOL)

        fpgs = 0
        feas_pgs = True
        offset = 0
        for k, s in structures.items():
            nu0 = nu_offset.get(k, 0)
            rng = slice(offset, offset + block_sizes[k])
            a, b, c, d, e = map(lambda v: v[rng], obj_voxels.arrays[1:-1])
            if isinstance(s, ro.structure.Target):
                fpgs += np.dot(d, nu[k] + nu0)
                feas_pgs &= np.all(a * nu[k] - b > -FEAS_TOL)
                feas_pgs &= np.all(a * nu[k] - b < 1 + FEAS_TOL)
            else:
                fpgs += 0
                feas_pgs &= np.all(abs(a * nu[k] - b) < FEAS_TOL)
            offset += block_sizes[k]

        assert abs(fstar_py(nu) - fpgs) < TOL
        assert feas_domain == feas_pgs
        return True

    wnorm = structures['target'].size
    opts = dict(weight_norm=wnorm, dose_norm=2.)
    nrm_opts = dict(target=opts, oar=opts)
    assert dual_objective_consistent()
    assert dual_objective_consistent(norm_options=dict(weight_norm=wnorm))
    assert dual_objective_consistent(norm_options=nrm_opts)
    assert dual_objective_consistent(
            voxel_weights=voxel_weights, norm_options=nrm_opts)
    assert dual_objective_consistent(nu_offset=nu_offset)
    assert dual_objective_consistent(
            nu_offset=nu_offset, voxel_weights=voxel_weights, norm_options=nrm_opts)
