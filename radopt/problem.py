import numpy as np
import collections

import radopt.structure
import radopt.primal
import radopt.dual

class Problem(object):
    def __init__(self, **options):
        self.structures = dict()
        self._global_dose_norm = 1.
        self._weight_norms = dict()

        self.structures.update(options.get('structures', dict()))

    def name_to_id(self, name):
        if name in self.structures:
            return name

        for sid in self.structures:
            if self.structures[sid].name == name:
                return sid

        raise KeyError('no structure with name/id `{}`'.format(name))

    def rekey_dictionary_by_ids(self, dictionary):
        return {self.name_to_id(k): dictionary[k] for k in dictionary}

    def joint_normalization(self, key_structure=None, key_wt=None,
                            key_dose=None, **options):
        if key_structure is None:
            return dict()

        key_id = self.name_to_id(key_structure)

        assert key_id in self.structures, "Problem has structure with ID {}".format(k)

        s_prime = self.structures[key_id]
        global_wt_norm = 1.
        global_dose_norm = 1.

        quotient = s_prime.size if options.pop('normalize_sizes', False) else 1
        if key_wt is not None:
            global_wt_norm = s_prime.objective.parameters[key_wt]
        if key_dose is not None:
            global_dose_norm = s_prime.objective.parameters[key_dose]
        self._global_dose_norm = global_dose_norm

        norm_options = dict()

        for sid in self.structures:
            s = self.structures[sid]
            size_s = 1. if s.collapsable(**options) else s.size
            norm_options[sid] = dict()
            norm_options[sid]['weight_norm'] = global_wt_norm * float(size_s)/quotient
            norm_options[sid]['dose_norm'] = global_dose_norm
            self._weight_norms[sid] = norm_options[sid]['weight_norm']

        return norm_options

    def normalize_problem(self, **options):
        """
        Tune problem scaling before handing off to solver.

        Default behavior is heuristic to obtain a well-conditioned problem.
        """
        def largest_target():
            key = None
            size = 0
            for k, s in self.structures.items():
                if isinstance(s, radopt.structure.Target) and s.size > size:
                    size = s.size
                    key = k
            return key
        opts=dict(key_structure=options.pop('key_structure', largest_target()))
        key_weight = options.pop('key_weight', 'weight_underdose')
        key_dose = options.pop('key_dose', 'dose')
        if options.pop('normalize_weights', False):
            opts['key_wt'] = key_weight
        if options.pop('normalize_doses', True):
            opts['key_dose'] = key_dose
        opts.update(options)
        options['norm_options'] = self.joint_normalization(**opts)
        return options

    def calc_row_dim(self, dose_matrices, **options):
        assert all((k in dose_matrices for k in self.structures)), (
                "Structure dose matrices assigned")

        block_sizes = collections.OrderedDict()
        for sid in self.structures:
            s = self.structures[sid]
            A = dose_matrices[sid]
            block_sizes[sid] = 1 if s.collapsable(**options) else A.shape[0]

        m = sum(block_sizes.values())
        return m, block_sizes

    def calc_col_dim(self, dose_matrices):
        assert len(set(A.shape[1] for A in dose_matrices.values())) == 1, (
                "Structure dose matrices have consistent # columns")

        return dose_matrices.values()[0].shape[1]

    def set_key_order(self, key_order=None, **options):
        if key_order is not None:
            keys = list(map(self.name_to_id, key_order))
            self.structures = collections.OrderedDict(
                    [(k, self.structures[k]) for k in keys])
        elif not isinstance(self.structures, collections.OrderedDict):
            self.structures = collections.OrderedDict(sorted(self.structures.items()))

    def build_primal_objectives(self, dose_matrices, **options):
        self.set_key_order(**options)
        m, block_sizes = self.calc_row_dim(dose_matrices, **options)
        n = self.calc_col_dim(dose_matrices)
        options = self.normalize_problem(**options)
        return radopt.primal.build_objectives(
                (m, n), block_sizes, self.structures, dose_matrices, **options)

    def build_dual_objectives(self, dose_matrices, **options):
        self.set_key_order(**options)
        m, block_sizes = self.calc_row_dim(dose_matrices, **options)
        n = self.calc_col_dim(dose_matrices)
        options = self.normalize_problem(**options)
        return radopt.dual.build_objectives(
                (m, n), block_sizes, self.structures, dose_matrices, **options)

    def working_form(self, dose_matrices, **options):
        self.set_key_order(**options)
        _, block_sizes = self.calc_row_dim(dose_matrices, **options)
        working_matrices = collections.OrderedDict()
        for sid in block_sizes:
            A_sub = dose_matrices[sid]
            if block_sizes[sid] == 1 and A_sub.shape[0] != 1:
                if A_sub.shape[0] != A_sub.size:
                    A_sub = A_sub.mean(axis=0)
                A_sub = A_sub.reshape((1, -1))
            working_matrices[sid] = A_sub
        return working_matrices

    def build_matrix(self, dose_matrices, **options):
        working_matrices = self.working_form(dose_matrices, **options)
        m = sum((working_matrices[k].shape[0] for k in working_matrices))
        n = self.calc_col_dim(dose_matrices)
        A = np.zeros((m, n))

        ptr = 0
        for sid in working_matrices:
            offset = working_matrices[sid].shape[0]
            A[ptr : ptr + offset, :] += working_matrices[sid]
            ptr += offset
        return A, working_matrices

    def contig_to_blocks(self, array, block_sizes):
        blocked_array = dict()
        ptr = 0
        for blk_id in block_sizes:
            size = block_sizes[blk_id]
            blocked_array[blk_id] = array[ptr : ptr + size, ...]
            ptr += size
        return blocked_array

    def blocks_to_contig(self, dictionary, transpose=False):
        ordered_arrays = tuple(dictionary[k] for k in self.structures)
        transpose |= all(len(A.shape) == 1 for A in ordered_arrays)
        if transpose:
            return np.hstack((A.T for A in ordered_arrays))
        else:
            return np.vstack(ordered_arrays)

    def current_weights(self, include_doses=False):
        output = dict()
        for sid in self.structures:
            params = self.structures[sid].objective.parameters
            for pid in params:
                if 'weight' in pid or include_doses:
                    output[sid, pid] = params[pid]
        return output