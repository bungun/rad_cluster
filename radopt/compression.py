""" TODO: docstring
"""
from __future__ import print_function
import numpy as np
import optkit as ok

from radopt import clustering
from radopt import planning
import radopt.dual

class ClusteredProblem(planning.IntensityOptimizationProblem):
    """ TODO: docstring
    """
    def __init__(self, **options):
        planning.IntensityOptimizationProblem.__init__(self, **options)
        self.A.update(options.get('A_full', options.pop('A', dict())))

        self._full_problem = planning.IntensityOptimizationProblem(**options)
        self._clus_problem = planning.IntensityOptimizationProblem(**options)

    @property
    def A_full(self):
        return self.A

    def full_problem(self):
        self._full_problem.A.update(self.A_full)
        self._full_problem.structures.update(self.structures)
        return self._full_problem

class VoxelClusteredProblem(ClusteredProblem):
    """ TODO: docstring
    """
    def __init__(self, **options):
        ClusteredProblem.__init__(self, **options)
        self.cluster_maps = dict()
        self.A_vclu = dict()
        self.voxel_weights = dict()

        self.A_vclu.update(options.pop('A_vclu', dict()))
        self.voxel_weights.update(options.pop('voxel_weights', dict()))
        self.cluster_maps.update(options.pop('cluster_maps', dict()))

    def build_solver(self, **options):
        """ TODO: docstring
        """
        s_keys = set(self.structures.keys())
        assert set(self.A_full.keys()) ==  s_keys
        assert set(self.A_vclu.keys()) == s_keys
        assert set(self.cluster_maps.keys()) == s_keys
        assert all(
                isinstance(m, clustering.DiscreteMap)
                for m in self.cluster_maps.values())

        self._full_problem.A.update(self.A_full)
        self._full_problem.structures.update(self.structures)
        self._full_problem.working_matrix(**options)

        for k in self.structures:
            if isinstance(self.cluster_maps[k], clustering.ClusterMap):
                self.voxel_weights[k] = self.cluster_maps[k].cluster_weights

        self._clus_problem.A.update(self.A_vclu)
        self._clus_problem.structures.update(self.structures)
        return self._clus_problem.build_solver(**options)

    def rescale_dual_vec(self, dual_vec):
        dual = self._clus_problem.to_blocks(dual_vec)
        rescaled = clustering.upsample(self.cluster_maps, dual, normalize=True)
        return self._full_problem.to_contig(rescaled)

    def solve_and_bound(self, solver, **options):
        """ TODO: docstring
        """
        f_full, fconj_full = self._full_problem.objectives(**options)
        # f_clu, fconj_clu = self._clus_problem.objectives(**options)
        solution = self._clus_problem.solve(
                solver,
                voxel_weights=self.voxel_weights,
                **options)

        x = solution['beam_weights']
        nu_clu = solution['voxel_duals']
        nu = self.rescale_dual_vec(nu_clu)

        solution['lower_bound'] = -fconj_full(nu)
        solution['upper_bound'] = f_full(x)
        solution['voxel_duals'] = nu
        return solution

class BeamClusteredProblem(ClusteredProblem):
    """ TODO: docstring
    """
    def __init__(self, **options):
        ClusteredProblem.__init__(self, **options)
        self.cluster_map = options.get('cluster_map', None)
        self.A_bclu = dict()

        self.A_bclu.update(options.pop('A_bclu', dict()))

    def build_solver(self, **options):
        """ TODO: docstring
        """
        s_keys = set(self.structures.keys())
        assert set(self.A_full.keys()) == s_keys
        assert set(self.A_bclu.keys()) == s_keys
        assert isinstance(self.cluster_map, clustering.DiscreteMap)

        self._full_problem.A.update(self.A_full)
        self._full_problem.structures.update(self.structures)
        self._full_problem.working_matrix(**options)

        self._clus_problem.A.update(self.A_bclu)
        self._clus_problem.structures.update(self.structures)
        return self._clus_problem.build_solver(**options)

    def dual_subproblem(self, solution, nu_t, cone_tol, **options):
        """ Solve dual feasibility subproblem:

            min  -f_conj(nu_offset + delta)
            s.t. A'(nu_offset + delta) = mu
                 mu >= 0
                 delta >= 0
                 nu_offset + delta \in dom(f_conj)

            introduce variable muu = mu - A'nu_offset. Problem is then

            min  -f_conj(nu_offset + delta)
            s.t. A'(delta) = muu
                 muu + mu_offset >= 0
                 delta >= 0
                 nu_offset + delta \in dom(f_conj) - nu

            with mu_offset = A'nu_offset
        """
        A_full = self._full_problem.A_working
        verbose_dual = options.get('verbose_dual', True)

        mu = radopt.dual.A_transpose_nu(A_full, nu_t)
        mu_infeas = mu * (mu < -abs(cone_tol))
        n_infeas = sum(mu_infeas < 0)
        if verbose_dual:
            print('# INFEASIBLE: {}/{}'.format(n_infeas, mu.size))

        dim_clu = self.cluster_map.n_clusters
        dim_beams_sub = min(dim_clu, n_infeas)
        if verbose_dual:
            print('SUBPROBLEM BEAMS: {}'.format(dim_beams_sub))
        # assert dim_beams_sub > 0, "require problem still infeasible"
        if dim_beams_sub == 0:
            return solution

        indices_subproblem = np.argsort(mu_infeas)[:dim_beams_sub]
        A_subproblem = {k: A_full[k][:, indices_subproblem] for k in A_full}

        subprob = planning.IntensityOptimizationProblem()
        subprob.structures.update(self.structures)
        subprob.A.update(A_subproblem)
        A_sub, A_sub_working = subprob.build_matrix(**options)

        # A_sub, A_sub_working = self.build_matrix(A_subproblem)
        if A_sub.shape[0] == A_sub.size:
            A_sub = A_sub.reshape((-1, 1))

        mu_offset = radopt.dual.A_transpose_nu(A_sub_working, nu_t)
        mu_offset += float(options.pop('epsilon', 0))

        dual_obj_voxels, dual_obj_beams, _, _ = subprob.build_dual_objectives(
                A_sub_working, mu_offset=mu_offset, nu_offset=nu_t, **options)
        delta = 0.
        with ok.api.PogsSolver(A_sub.T) as dual_solver:
            dual_solver.solve(dual_obj_beams, dual_obj_voxels, **options)
            delta = dual_solver.output.x
            solution['dual_setup_time'] += dual_solver.info.setup_time
            solution['dual_solve_time'] += dual_solver.info.solve_time
            solution['voxel_duals'] += delta
        return solution

    def solve_and_bound(self, solver, **options):
        """ TODO: docstring
        """
        verbose_dual = options.get('verbose_dual', True)
        compression = float(options.get('compression', 1))

        f, fconj = self._full_problem.objectives(**options)
        feasible = self._full_problem.dual_conditions(**options)

        solution = self._clus_problem.solve(solver, **options)
        solution['beam_weights'] = self.cluster_map.upsample(
                solution['beam_weights'], normalize=True)
        solution['lower_bound'] = 0.
        solution['dual_setup_time'] = 0.
        solution['dual_solve_time'] = 0.

        p_UB = solution['upper_bound']
        p_LB = solution['lower_bound']
        nu_initial = self._full_problem.to_blocks(solver.output.nu)
        mu_initial = radopt.dual.A_transpose_nu(
                self._full_problem.A_working, nu_initial)
        mu_tol = abs(radopt.dual.A_transpose_nu(
                self._clus_problem.A_working, nu_initial).min()) * compression

        domain_tol = options.pop(
                'domain_tolerance', options.pop('dual_tolerance', 1e-2))
        cone_tol = options.pop('cone_tolerance', min(domain_tol, mu_tol))
        dual_tols = (domain_tol, cone_tol)

        nu_t = nu_initial
        in_domain, in_cone = feasible(nu_t, dual_tols, verbose=verbose_dual)
        if in_domain and in_cone:
            if verbose_dual:
                print('feasible after compressed solve')
            p_LB = max(0, -fconj(nu_t))


        max_epochs = int(np.ceil(np.true_divide(
                self.cluster_map.n_points, self.cluster_map.n_clusters)))

        epoch = 0
        while epoch < max_epochs and not in_cone:
            epoch += 1
            solution = self.dual_subproblem(solution, nu_t, cone_tol, **options)
            nu_t = self._full_problem.to_blocks(solution['voxel_duals'])
            p_LB = max(p_LB, -fconj(nu_t))
            in_domain, in_cone = feasible(nu_t, dual_tols)
            if verbose_dual:
                print('DUAL EPOCH: {}\t LOWER BOUND: {}\t UPPER BOUND: {}'.format(
                        epoch, p_LB, p_UB))
                print('in domain: {}'.format(in_domain))
                print('in cone: {}'.format(in_cone))

        if verbose_dual:
            print('feasible after {} dual solve epochs'.format(epoch))
        solution['dual_epochs'] = epoch
        solution['lower_bound'] = p_LB
        return solution

class CompressionAnalysis(object):
    @staticmethod
    def true_error(compressed_solutions, full_solutions):
        for key in compressed_solutions:
            if key in full_solutions:
                ub = compressed_solutions[key]['upper_bound']
                pstar = full_solutions[key]['pstar']
                true_error = 100 * (ub - pstar) / ub
                compressed_solutions[key]['true_error'] = true_error
        return compressed_solutions

    @staticmethod
    def suboptimality_bound(compressed_solutions):
        for key in compressed_solutions:
            ub = compressed_solutions[key]['upper_bound']
            lb = compressed_solutions[key]['lower_bound']
            bound = 100 * (ub - lb) / ub
            compressed_solutions[key]['suboptimality_bound'] = bound
        return compressed_solutions

    @staticmethod
    def analyze(compressed_solutions, full_solutions=None, key_cold='nominal'):
        compressed_solutions = CompressionAnalysis.suboptimality_bound(
                compressed_solutions)
        if full_solutions is None:
            mutual_keys = set(compressed_solutions.keys())
        else:
            compressed_solutions = CompressionAnalysis.true_error(
                    compressed_solutions, full_solutions)
            mutual_keys = set(compressed_solutions.keys()) & set(full_solutions.keys())

        metadata = dict(shape=None, warm_start_runs=len(mutual_keys)-1)
        cold_start = dict(
                primal_time=None,
                dual_time=None,
                suboptimality=None,
                true_error=None)
        warm_start = dict(
                primal_times=list(),
                dual_times=list(),
                suboptimalities=list(),
                true_errors=list())

        for k in mutual_keys:
            sol = compressed_solutions[k]
            pt = sol.get('solve_time') + sol.get('setup_time')
            dt = sol.get('dual_solve_time', 0.) + sol.get('dual_setup_time', 0.)
            so = sol.get('suboptimality_bound')
            te = sol.get('true_error')
            if k == key_cold:
                metadata['shape'] = sol['dimension']
                cold_start['suboptimality'] = so
                cold_start['true_error'] = te
                cold_start['primal_time'] = pt
                cold_start['dual_time'] = dt
            else:
                warm_start['suboptimalities'].append(so)
                warm_start['true_errors'].append(te)
                warm_start['primal_times'].append(pt)
                warm_start['dual_times'].append(dt)

        def calc_stats(tag, series, **stats):
            try:
                output = dict()
                for name, stat in stats.items():
                    output[name + "_" + tag] = stat(series)
                return output
            except:
                return dict()

        stats = dict(mean=np.mean, maximum=np.max, median=np.median)
        warm_start.update(calc_stats('primal_time', warm_start.pop('primal_times'), **stats))
        warm_start.update(calc_stats('dual_time', warm_start.pop('dual_times'), **stats))
        warm_start.update(calc_stats('suboptimality', warm_start.pop('suboptimalities'), **stats))
        warm_start.update(calc_stats('true_error', warm_start.pop('true_errors'), **stats))

        return dict(cold=cold_start, warm=warm_start, meta=metadata)

