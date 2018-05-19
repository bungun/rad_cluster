import collections
import optkit as ok
from radopt import problem

class IntensityOptimizationProblem(problem.Problem):
    """ TODO: docstring """
    def __init__(self, **options):
        problem.Problem.__init__(self, **options)
        self.A_working = dict()
        self.A = dict()
        self.block_sizes = collections.OrderedDict()

        self.A.update(options.get('A', dict()))

    def working_matrix(self, **options):
        """
        Dictionary representation of working forms of problem matrices
        """
        A_dict = problem.Problem.working_form(self, self.A, **options)
        self.A_working.update(A_dict)
        self.block_sizes.update({k: A_dict[k].shape[0] for k in self.structures})
        return A_dict

    def build_matrix(self, **options):
        """
        Concatenated and dictionary representations of working forms of problem matrices.
        """
        A, A_dict =  problem.Problem.build_matrix(self, self.A, **options)
        self.A_working.update(A_dict)
        self.block_sizes.update({k: A_dict[k].shape[0] for k in self.structures})
        return A, A_dict

    def build_solver(self, **options):
        """ TODO: docstring """
        n_structures = len(self.structures)
        assert set(self.A.keys()) == set(self.structures.keys()), (
                'dose matrix assigned for each structure')

        A, _ = self.build_matrix(**options)
        return ok.api.PogsSolver(A)

    def objectives(self, **options):
        options['normalize_doses'] = False
        _, _, f_eval = self.build_primal_objectives(self.A_working, **options)
        _, _, fconj_eval, _ = self.build_dual_objectives(self.A_working, **options)
        return f_eval, fconj_eval

    def dual_conditions(self, **options):
        _, _, _, feas = self.build_dual_objectives(self.A_working, **options)
        return feas

    def solve(self, solver, **options):
        """ TODO: docstring """
        f, fconj = self.objectives(**options)
        obj_voxels, obj_beams, _ = self.build_primal_objectives(self.A_working, **options)
        dose_norm = self._global_dose_norm

        solver.solve(obj_voxels, obj_beams, **options)
        x_star = solver.output.x * dose_norm
        nu_star = 1. * solver.output.nu
        pstar = f(x_star)
        dstar = -fconj(nu_star)

        return dict(
                pstar=pstar,
                dstar=dstar,
                upper_bound=pstar,
                lower_bound=dstar,
                beam_weights=x_star,
                voxel_duals=nu_star,
                rho=solver.info.rho,
                setup_time=solver.info.setup_time,
                solve_time=solver.info.solve_time,
                dimension=solver.shape,
                dose_norm=dose_norm)

    def solve_and_bound(self, solver, **options):
        return self.solve(solver, **options)

    def to_blocks(self, array):
        assert self.block_sizes, ('block sizes known')
        return self.contig_to_blocks(array, self.block_sizes)

    def to_contig(self, blocks):
        return self.blocks_to_contig(blocks)
