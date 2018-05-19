import numpy as np
import optkit as ok
import radopt as ro
import unittest

class VoxelClusteringTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.shape = self.voxels, self.beams = voxels, beams = 100, 50
        self.ovox = ovox = int(0.4 * voxels)
        self.tvox = tvox = voxels - ovox
        self.tvox_compressed = tvox_compressed = 20

        self.structures = dict(
                target=ro.structure.Target('target', tvox, dose=75),
                oar=ro.structure.OAR('oar', ovox))

        def gen_u(n, k):
            """
            generate cluster map where every cluster guaranteed to appear at least once
            """
            return np.hstack((
                    np.random.permutation(k),
                    np.random.randint(0, k, n-k)))

        u = gen_u(tvox, tvox_compressed)

        self.cluster_maps = cluster_maps = dict(
            target=ro.clustering.ClusterMap(tvox, tvox_compressed, u),
            oar=ro.clustering.IdentityMap())

        self.A_vclu = A_vclu = dict(
                target=np.random.uniform(0, 4, (tvox_compressed, beams)),
                oar=np.random.uniform(0, 1, (ovox, beams)))
        self.A_full = A_full = dict(
                target=cluster_maps['target'].upsample(A_vclu['target']),
                oar=A_vclu['oar'])

        self.voxel_weights = voxel_weights = dict(
                target=cluster_maps['target'].cluster_weights)

    def test_exact_separate_problems(self):
        r"""
        exact variant :math:`A_{full} == UA_{vclu}` as two separate optimization problems
        """
        prb = ro.planning.IntensityOptimizationProblem()
        prb_clu = ro.planning.IntensityOptimizationProblem()
        prb.structures.update(self.structures)
        prb_clu.structures.update(self.structures)
        prb.A.update(self.A_full)
        prb_clu.A.update(self.A_vclu)
        solver = prb.build_solver()
        solver_clu = prb_clu.build_solver()
        solution = prb.solve(solver)
        solution_clu = prb_clu.solve(solver_clu, voxel_weights=self.voxel_weights)

        pstar = solution.get('upper_bound')
        pstar_clu = solution_clu.get('upper_bound')
        dstar = solution.get('lower_bound')
        dstar_clu = solution_clu.get('lower_bound')
        dim = np.sqrt(solver_clu.A.size)
        ptol, dtol = 1e-2 * (dim + pstar), 1e-3 * (dim + dstar)

        # objective values close
        assert abs(pstar - pstar_clu) < ptol, '|{} - {}| = {} < {}'.format(
                pstar, pstar_clu, pstar-pstar_clu, ptol)
        assert abs(dstar - dstar_clu) < dtol, '|{} - {}| = {} < {}'.format(
                dstar, dstar_clu, dstar-dstar_clu, dtol)

        # solutions close
        dimx, normx = np.log(solver.output.x.size), np.linalg.norm(solver.output.x)
        norm_diff = np.linalg.norm(solver.output.x - solver_clu.output.x)
        assert norm_diff < 5e-3 * (dimx + normx), '|x-x_clu| = {} < {}'.format(norm_diff, normx)

    def test_exact(self):
        r"""
        exact variant :math:`A_{full} == UA_{vclu}` as voxel-clustered problem
        """
        prb = ro.compression.VoxelClusteredProblem()
        prb.structures.update(self.structures)
        prb.A_full.update(self.A_full)
        prb.A_vclu.update(self.A_vclu)
        prb.cluster_maps.update(self.cluster_maps)
        solver = prb.build_solver()
        solution = prb.solve_and_bound(solver)
        ub, lb = solution['upper_bound'], solution['lower_bound']
        ptol = 10 * (
                solver.settings.reltol * ub
                + solver.settings.abstol * np.sqrt(solver.A.size))
        assert np.any(solution['beam_weights'] > 0)

        # full primal eval == clustered primal eval for exact problem
        assert abs(ub - lb) < ptol, '|{} - {}| = {} < {}'.format(
                ub, lb, ub-lb, ptol)

    def test_exact_plus_noise(self):
        r""" exact + noise variant

        :math:`A_{full} = UA_{vclu} + \varepsilon \\
              \varepsilon \sim \mathcal{N}(0, 1/b)^{m \times n},\quad b > 0`
        """
        tvox, tvox_compressed, ovox, beams = (
                self.tvox, self.tvox_compressed, self.ovox, self.beams)
        structures, cluster_maps = self.structures, self.cluster_maps

        A_vclu = dict(
                target=np.random.uniform(0, 4, (tvox_compressed, beams)),
                oar=np.random.uniform(0, 1, (ovox, beams)))
        A_full = dict(
                target=cluster_maps['target'].upsample(A_vclu['target']) + np.random.normal(0, 0.1, (tvox, beams)),
                oar=A_vclu['oar'])

        prb = ro.compression.VoxelClusteredProblem()
        prb.structures.update(structures)
        prb.A_full.update(A_full)
        prb.A_vclu.update(A_vclu)
        prb.cluster_maps.update(cluster_maps)
        solver = prb.build_solver()
        solution = prb.solve_and_bound(solver)

        ub, lb = solution['upper_bound'], solution['lower_bound']
        assert np.any(solution['beam_weights'] > 0)

        # full primal eval >= clustered primal eval for non-exact problem
        assert ub >= lb, 'UB: {} > LB {}'.format(ub, lb)


    def test_vclu(self):
        r""" test inexact (usual) variant :math:`A_{full} \approx UA_{vclu}`
        """
        tvox, tvox_compressed, ovox, beams = (
                self.tvox, self.tvox_compressed, self.ovox, self.beams)
        structures, cluster_maps = self.structures, self.cluster_maps

        A_full = dict(
                target=np.random.uniform(0, 4, (tvox, beams)),
                oar=np.random.uniform(0, 1, (ovox, beams)))
        A_vclu = dict(
            target=cluster_maps['target'].downsample(A_full['target']),
            oar=A_full['oar'],)

        prb = ro.compression.VoxelClusteredProblem()
        prb.structures.update(structures)
        prb.A_full.update(A_full)
        prb.A_vclu.update(A_vclu)
        prb.cluster_maps.update(cluster_maps)
        solver = prb.build_solver()
        solution = prb.solve_and_bound(solver)
        assert np.any(solution['beam_weights'] > 0)
        ub, lb = solution['upper_bound'], solution['lower_bound']

        # full primal eval >= clustered primal eval for approximate problem
        assert ub >= lb, 'UB {} > LB {}'.format(ub, lb)

        yclu = ro.primal.A_mul_x(A_full, solution['beam_weights'])

        prb = ro.planning.IntensityOptimizationProblem()
        prb.structures.update(structures)
        prb.A.update(A_full)
        pstar = prb.solve(prb.build_solver())['pstar']

        assert ub >= pstar, 'UB: {} > p^\star {}'.format(ub, pstar)
        assert pstar >= lb, 'p^\star {} > LB {}'.format(pstar, lb)

class BeamClusteringTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        voxels, beams = 100, 200
        self.shape = self.voxels, self.beams = voxels, beams = 100, 200
        self.ovox = ovox = int(0.4 * voxels)
        self.tvox = tvox = voxels - ovox
        self.beams_compressed = beams_compressed = 20
        self.structures = structures = dict(
                target=ro.structure.Target('target', tvox, dose=75),
                oar=ro.structure.OAR('oar', ovox))

        def gen_v(n, k):
            """
            generate cluster map where every cluster guaranteed to appear at least once
            """
            return np.hstack((
                    np.random.permutation(k),
                    np.random.randint(0, k, n-k)))

        v = gen_v(beams, beams_compressed)
        self.cluster_map = cluster_map = ro.clustering.ClusterMap(
                beams, beams_compressed, v)

        A_bclu0 = dict(
                target=np.random.uniform(0, 4, (tvox, beams_compressed)),
                oar=np.random.uniform(0, 1, (ovox, beams_compressed)),
        )
        A_full = ro.clustering.upsample(cluster_map, A_bclu0, transpose=True)
        A_full['target'] += np.random.normal(0, 0.1, (tvox, beams))
        self.A_full = A_full
        self.A_bclu = ro.clustering.downsample(cluster_map, A_full, transpose=True)

    def test_dual_subproblem(self):
        tvox, ovox, beams, beams_compressed = (
                self.tvox, self.ovox, self.beams, self.beams_compressed)

        structures, cluster_map, A_full, A_bclu = (
                self.structures, self.cluster_map, self.A_full, self.A_bclu)

        prb = ro.planning.IntensityOptimizationProblem()
        prb_clu = ro.planning.IntensityOptimizationProblem()
        prb.structures.update(structures); prb_clu.structures.update(structures)
        prb.A.update(A_full); prb_clu.A.update(A_bclu)

        solver, solver_clu = prb.build_solver(), prb_clu.build_solver()

    #     ov, ob, f = prb.build_primal_objectives(prb.A_working)
        _, _, fconj, feasible = prb.build_dual_objectives(prb.A_working)

        solution, solution_clu = prb.solve(solver), prb_clu.solve(solver_clu)

        pstar, pstar_clu = solution['pstar'], solution_clu['pstar']
        UB = pstar_clu
        assert UB >= pstar, "UB > p^\star: {} > {}".format(UB, pstar)

        xstar = solution['beam_weights']
        xstar_clu = solution_clu['beam_weights']
        xstar_us = cluster_map.upsample(solution_clu['beam_weights'], normalize=True)

        nustar = solution['voxel_duals']
        nustar_clu = solution_clu['voxel_duals']
        mu = solver.A.T.dot(nustar)
        mu_clu = solver.A.T.dot(nustar_clu)

        cone_tol = abs(mu.min())
        domain_tol = 1e-2
        dual_tols = (domain_tol, cone_tol)

        assert all(feasible(nustar, dual_tols))
        assert not all(mu_clu > -abs(cone_tol))
        assert pstar > 0 and pstar_clu > 0
        assert pstar_clu > pstar

        def n_infeas(nu_vec):
            return sum(solver.A.T.dot(nu_vec) < -abs(cone_tol))
        infeas_before = n_infeas(nustar_clu)

        def dual_subproblem(solution, nu_t, dual_tols, **options):
            try:
                _, cone_tol = dual_tols
            except:
                cone_tol = dual_tols
            mu = ro.dual.A_transpose_nu(prb.A_working, nu_t)
            mu_infeas = mu * (mu < -abs(cone_tol))
            n_infeas = sum(mu_infeas < 0)
            print('# INFEASIBLE: {}/{}'.format(n_infeas, mu.size))

            dim_clu = cluster_map.n_clusters
            dim_beams_sub = min(dim_clu, n_infeas)
            print('SUBPROBLEM BEAMS: {}'.format(dim_beams_sub))
            assert dim_beams_sub > 0, "require problem still infeasible"

            subprob = ro.planning.IntensityOptimizationProblem()
            subprob.structures.update(prb.structures)

            indices_subproblem = np.argsort(mu_infeas)[:dim_beams_sub]
            subprob.A.update({
                    k: prb.A_working[k][:, indices_subproblem] for k in A_full})

            A_sub, A_sub_working = subprob.build_matrix()
            if A_sub.shape[0] == A_sub.size:
                A_sub = A_sub.reshape((-1, 1))

            mu_offset = ro.dual.A_transpose_nu(A_sub_working, nu_t)

            dual_obj_voxels, dual_obj_beams, _, _ = subprob.build_dual_objectives(
                    A_sub_working, mu_offset=mu_offset, nu_offset=nu_t)
            delta = 0.
            with ok.api.PogsSolver(A_sub.T) as dual_solver:
                dual_solver.solve(dual_obj_beams, dual_obj_voxels, **options)
                delta = dual_solver.output.x
                solution['dual_setup_time'] += dual_solver.info.setup_time
                solution['dual_solve_time'] += dual_solver.info.solve_time
                solution['voxel_duals'] += delta
            return solution

        solution_running = dict()
        solution_running.update(solution_clu)
        solution_running['dual_setup_time'] = 0.
        solution_running['dual_solve_time'] = 0.
        nu_t = prb.to_blocks(solution_running['voxel_duals'])
        p_LB = 0.
        p_UB = solution_running['upper_bound']
        in_domain, in_cone = feasible(nu_t, dual_tols)

        # 1 iteration
        epoch = 0
        while epoch < 1 and not (in_domain and in_cone):
            epoch += 1
            solution_running = dual_subproblem(solution_running, nu_t, dual_tols, **dict())
            nu_t = prb.to_blocks(solution_running['voxel_duals'])
            p_LB = max(p_LB, -fconj(nu_t))
            in_domain, in_cone = feasible(nu_t, dual_tols)
            print('DUAL EPOCH: {}\t LOWER BOUND: {}\t UPPER BOUND: {}'.format(
                    epoch, p_LB, p_UB))
            print('in domain: {}'.format(in_domain))
            print('in cone: {}'.format(in_cone))

        assert in_domain, 'nu_{} in domain'.format(epoch)
        infeas_after = n_infeas(solution_running['voxel_duals'])
        print 'infeas before, after:', infeas_before, infeas_after
        assert in_cone or (infeas_after < infeas_before)
        assert p_LB <= pstar and pstar <= p_UB, '{} <= {} <= {}'.format(p_LB, pstar, p_UB)

    def test_bclu(self):
        prb = ro.compression.BeamClusteredProblem()
        prb.structures.update(self.structures)
        prb.A_full.update(self.A_full)
        prb.A_bclu.update(self.A_bclu)
        prb.cluster_map = self.cluster_map
        solver = prb.build_solver()
        solution = prb.solve_and_bound(solver)
        assert np.any(solution['beam_weights'] > 0)
        ub, lb = solution['upper_bound'], solution['lower_bound']

        # bounds in order
        tol = (
                abs(ub) * 10 * solver.settings.reltol
                + np.log(solver.A.size) * solver.settings.abstol)
        assert ub - lb >= -tol, 'UB: {} > LB {}'.format(ub, lb)

        prb = ro.planning.IntensityOptimizationProblem()
        prb.structures.update(self.structures)
        prb.A.update(self.A_full)
        full_sol = prb.solve(prb.build_solver())
        pstar = max(full_sol['pstar'], full_sol['dstar'])

        # bound interval contains true optimum
        assert ub - pstar >= -tol, 'UB: {} > p^\star {}'.format(ub, pstar)
        assert pstar - lb >= -tol, 'p^\star {} > LB {}'.format(pstar, lb)

