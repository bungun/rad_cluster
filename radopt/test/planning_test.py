import collections
import numpy as np
import radopt as ro

def test_intensity_optimization_problem():
    voxels = 110
    beams = int(np.random.uniform(voxels/2, 2 * voxels))
    ovox = 10
    tvox = voxels - ovox
    tvox1 = int(0.6 * tvox)
    tvox2 = tvox - tvox1
    structures = {
            0: ro.structure.Target('target', tvox1),
            1: ro.structure.Target('target2', tvox2),
            2: ro.structure.OAR('oar', ovox)}
    structures[0].objective.parameters['weight_underdose'] = np.random.uniform(1, 2)
    structures[0].objective.parameters['dose'] = np.random.uniform(10, 20)
    dose_matrices = {k: np.random.random((structures[k].size, beams)) for k in structures}

    prb = ro.planning.IntensityOptimizationProblem()
    prb.structures.update(structures)
    prb.A.update(dose_matrices)

    solver = prb.build_solver()
    assert solver.shape == (tvox + 1, beams)
    del solver

    prb.build_primal_objectives(prb.A, normalize_weights=False, normalize_doses=True)
    gnorm = prb._global_dose_norm
    _, _, ff = prb.build_primal_objectives(prb.A, normalize_weights=False, normalize_doses=False)
    _, _, ffw = prb.build_primal_objectives(prb.A, normalize_weights=True, normalize_doses=False)
    iters = dict()
    objvals = list()

    for nw in (False, True):
        for nd in (False, True):
            solver = prb.build_solver()
            solution = prb.solve(solver, normalize_weights=nw, normalize_doses=nd, verbose=0)
            ub, lb = solution.get('upper_bound'), solution.get('lower_bound')
            assert ub >= 0 and lb >= 0
            tol = 1e-2 * ub + np.sqrt(solver.A.size) * 1e-3
            assert abs(ub - lb) < tol, '{} {} {}'.format(ub, lb, tol)
            iters[(nw, nd)] = solver.info.iterations
            nrm = gnorm if nd else 1
            objvals.append(ff(solver.output.x * nrm))
            del solver

    # scaling does not change optimal objective value
    assert all(abs(np.array(objvals) - objvals[0]) < 5e-2 * (1 + objvals[0]))

    iters = collections.OrderedDict(sorted(iters.items(), key=lambda t: t[1], reverse=True))
    print "weight normalized? | dose normalized ? | iters "
    print "-------------------|-------------------|-------"
    for nw, nd in iters:
        print "{!r:^19}|{!r:^19}|{:^7}".format(nw, nd, iters[(nw, nd)])
    print "\n(1 run each condition)"

    # dose normalization is always good
    assert iters[(False, True)] < iters[(False, False)]
    assert iters[(False, True)] < iters[(True, False)]
    assert iters[(True, True)] < iters[(False, False)]
    assert iters[(True, True)] < iters[(True, False)]

    # dose normalization alone usually fewer iterations than doubly-normalized
    iters = dict(dose_and_wt=0, dose_only=0)
    runs = 20
    for i in range(runs):
        dose_matrices = {k: np.random.random((structures[k].size, beams)) for k in structures}
        prb.A.update(dose_matrices)
        solver = prb.build_solver()
        for nw in (True, False):
            prb.solve(solver, normalize_doses=True, normalize_weights=nw, verbose=0)
            key = 'dose_and_wt' if nw else 'dose_only'
            iters[key] += solver.info.iterations / float(runs)

    print "\n\nnormalization policy? | iters "
    print "----------------------|-------"
    for policy in iters:
        print "{:^22}|{:^7.0f}".format(policy, iters[policy])
    print "\n(average of {} runs)".format(runs)

    assert iters['dose_only'] < iters['dose_and_wt']