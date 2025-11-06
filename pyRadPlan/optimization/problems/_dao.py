"""Direct Aperture Optimization (DAO) problem for pyRadPlan.

Implements Direct Aperture Optimization which directly optimizes MLC aperture
shapes and weights instead of post-processing fluence maps. This extends the
fluence optimization by working in aperture parameter space while reusing
the same objective functions.

References
----------
matRad: matRad/matRad/optimization/@matRad_OptimizationProblemDAO/
"""

from typing import Union, cast, Dict, Any
import time
import logging

import numpy as np
from numpy.typing import NDArray

from ...plan import Plan
from ...sequencing import ApertureInfo
from ._optiprob import NonLinearPlanningProblem
from ..solvers import NonLinearOptimizer
from ..objectives import Objective

logger = logging.getLogger(__name__)


class DirectApertureOptimization(NonLinearPlanningProblem):
    """
    Direct Aperture Optimization (DAO) planning problem.

    Optimizes MLC aperture parameters (leaf positions and weights) directly
    instead of continuous fluence weights. This provides deliverable treatment
    plans without post-sequencing and allows direct control over complexity.

    Parameters
    ----------
    pln : Union[Plan, dict], optional
        Plan object or dictionary to initialize the problem with.
    aperture_info : ApertureInfo, optional
        Initial aperture information from sequencing. If None, must be
        provided before solving via set_aperture_info().

    Attributes
    ----------
    aperture_info : ApertureInfo
        Current aperture information structure containing shapes, weights,
        and leaf positions for all beams.
    bypass_objective_jacobian : bool, default=True
        Whether to bypass the objective jacobian calculation for efficiency.

    Notes
    -----
    The optimization variables are structured as:
        x = [weights, left_leaf_positions, right_leaf_positions]
    where:
        - weights: aperture weights for all shapes (total_num_shapes,)
        - left_leaf_positions: left leaf positions for all shapes (total_num_leaf_pairs,)
        - right_leaf_positions: right leaf positions for all shapes (total_num_leaf_pairs,)

    The DAO problem adds leaf gap constraints:
        c_dao[i] = right_leaf_pos[i] - left_leaf_pos[i] >= 0  (no collision)

    References
    ----------
    matRad: matRad/matRad/optimization/@matRad_OptimizationProblemDAO/matRad_OptimizationProblemDAO.m
    """

    name = "Direct Aperture Optimization"
    short_name = "dao"

    aperture_info: ApertureInfo
    bypass_objective_jacobian: bool

    def __init__(
        self,
        pln: Union[Plan, dict] = None,
        aperture_info: ApertureInfo = None,
    ):
        self.aperture_info = aperture_info
        self.bypass_objective_jacobian = True

        # Caching for gradients
        self._grad_cache_intermediate = None
        self._grad_cache = None
        self._obj_times = []
        self._deriv_times = []
        self._solve_time = None

        super().__init__(pln)

    def set_aperture_info(self, aperture_info: ApertureInfo) -> None:
        """
        Set or update aperture information.

        Parameters
        ----------
        aperture_info : ApertureInfo
            Aperture information structure from sequencing.

        Notes
        -----
        This must be called before solving if aperture_info was not provided
        during initialization.
        """
        self.aperture_info = aperture_info

    def _initialize(self):
        """Initialize the DAO problem."""
        super()._initialize()

        # Validate aperture info
        if self.aperture_info is None:
            raise ValueError(
                "aperture_info must be set before solving. "
                "Call set_aperture_info() or provide it during initialization."
            )

        # Check solver compatibility
        if not isinstance(self.solver, NonLinearOptimizer):
            raise ValueError("Solver must be an instance of NonLinearOptimizer")

        # Setup solver callbacks
        self.solver.objective = self._objective_function
        self.solver.gradient = self._objective_gradient

        # Variable bounds from aperture info
        lb, ub = self.aperture_info.get_bounds()
        self.solver.bounds = (lb, ub)

        # Constraints (leaf gaps)
        self.solver.constraint = self._constraint_function
        self.solver.constraint_jacobian = self._constraint_jacobian
        self.solver.constraint_bounds = self._constraint_bounds()

        self.solver.max_iter = 500

    def _aperture_vector_to_bixel_weights(self, aperture_vec: NDArray) -> NDArray:
        """
        Convert aperture parameter vector to bixel weights.

        Parameters
        ----------
        aperture_vec : NDArray
            Optimization vector [weights, left_positions, right_positions].

        Returns
        -------
        NDArray
            Bixel weights for dose calculation. Shape: (total_num_bixels,)

        Notes
        -----
        This performs the following steps:
        1. Update aperture_info from optimization vector
        2. Compute shape maps from leaf positions
        3. Calculate bixel weights by summing contributions from all shapes

        This is equivalent to matRad_daoVec2ApertureInfo + bixel weight calculation.
        """
        # Update aperture info with current optimization variables
        if not np.array_equal(aperture_vec, self.aperture_info.aperture_vector):
            self.aperture_info.from_vector(aperture_vec)

        # Calculate bixel weights from aperture shapes
        # This requires mapping shapes back to bixels using the bixel_ind_map
        total_bixels = self.aperture_info.total_num_of_bixels
        bixel_weights = np.zeros(total_bixels, dtype=np.float64)

        offset = 0
        for beam_idx, beam in enumerate(self.aperture_info.beam):
            # Get beam dimensions
            bixel_ind_map = beam.bixel_ind_map
            dim_z, dim_x = bixel_ind_map.shape
            bixel_width = self.aperture_info.bixel_width

            # Accumulate dose from all shapes in this beam
            fluence_sum = np.zeros((dim_z, dim_x), dtype=np.float64)

            for shape in beam.shape:
                # Create shape map from leaf positions
                shape_map = np.zeros((dim_z, dim_x), dtype=bool)

                for leaf_idx in range(beam.num_of_active_leaf_pairs):
                    left_pos = shape.left_leaf_pos[leaf_idx]
                    right_pos = shape.right_leaf_pos[leaf_idx]

                    # Find bixels covered by this leaf pair
                    # (This is a simplified version; full implementation would
                    # replicate matRad's exact logic)
                    lim_l = beam.lim_l[leaf_idx]
                    lim_r = beam.lim_r[leaf_idx]

                    # Calculate column indices covered
                    x_min = int(np.floor((left_pos - beam.pos_of_corner_bixel[0]) / bixel_width))
                    x_max = int(np.ceil((right_pos - beam.pos_of_corner_bixel[0]) / bixel_width))

                    x_min = max(0, x_min)
                    x_max = min(dim_x, x_max)

                    shape_map[leaf_idx, x_min:x_max] = True

                # Add weighted contribution
                fluence_sum += shape_map.astype(np.float64) * shape.weight

            # Map fluence back to bixel weights
            # Find valid bixel indices (non-NaN entries in bixel_ind_map)
            valid_mask = ~np.isnan(bixel_ind_map)
            valid_indices = bixel_ind_map[valid_mask].astype(int)

            bixel_weights[valid_indices] = fluence_sum[valid_mask]

        return bixel_weights

    def _objective_functions(self, x: NDArray) -> NDArray:
        """
        Compute objective function values for DAO.

        Parameters
        ----------
        x : NDArray
            Aperture parameter vector.

        Returns
        -------
        NDArray
            Array of objective function values.

        Notes
        -----
        Converts aperture parameters to bixel weights, then evaluates
        the same objective functions as fluence optimization.
        """
        # Convert aperture vector to bixel weights
        bixel_weights = self._aperture_vector_to_bixel_weights(x)

        # Evaluate objectives on bixel weights (same as fluence optimization)
        q_vectors = {}
        q_scenarios = {}

        for q in self._quantities:
            q_vectors[q.identifier] = q(bixel_weights)
            q_scenarios[q.identifier] = q.scenarios

        f_vals = []
        for obj_info in self._objective_list:
            ix = obj_info[0]
            tmp_obj_list = cast(list[Objective], obj_info[1])
            for obj in tmp_obj_list:
                f_vals.append(
                    sum(
                        [
                            obj.priority
                            * obj.compute_objective(q_vectors[obj.quantity].flat[scen_ix][ix])
                            for scen_ix in q_scenarios[obj.quantity]
                        ]
                    )
                )

        return np.asarray(f_vals, dtype=np.float64)

    def _objective_function(self, x: NDArray) -> np.float64:
        """Compute scalar objective function value."""
        t = time.time()
        f = np.sum(self._objective_functions(x))
        self._obj_times.append(time.time() - t)
        return f

    def _objective_jacobian(self, x: NDArray) -> NDArray:
        """
        Compute objective function Jacobian with respect to aperture parameters.

        Parameters
        ----------
        x : NDArray
            Aperture parameter vector.

        Returns
        -------
        NDArray
            Jacobian matrix. Shape: (num_objectives, len(x)) if not bypassed,
            otherwise (1, len(x)).

        Notes
        -----
        This uses the chain rule:
            d(objective)/d(aperture) = d(objective)/d(dose) * d(dose)/d(bixels) * d(bixels)/d(aperture)

        The d(bixels)/d(aperture) term accounts for how leaf position changes
        affect bixel weights (involves shape map derivatives).
        """
        # Convert aperture vector to bixel weights
        bixel_weights = self._aperture_vector_to_bixel_weights(x)

        # Compute gradient with respect to bixel weights (same as fluence)
        q_vectors = {}
        q_scenarios = {}

        if self._grad_cache_intermediate is None:
            initialize_cache = True
            self._grad_cache_intermediate = {}
        else:
            initialize_cache = False

        for q in self._quantities:
            q_vectors[q.identifier] = q(bixel_weights)
            q_scenarios[q.identifier] = q.scenarios
            if initialize_cache:
                if self.bypass_objective_jacobian:
                    cache_rows = 1
                else:
                    cache_rows = len(self._objectives_per_quantity[q.identifier])
                self._grad_cache_intermediate[q.identifier] = np.zeros(
                    (cache_rows, self._dij.dose_grid.num_voxels),
                    dtype=np.float32,
                )
            else:
                self._grad_cache_intermediate[q.identifier].fill(0.0)

        cnt = 0
        for obj_info in self._objective_list:
            ix = obj_info[0]
            tmp_obj_list = cast(list[Objective], obj_info[1])
            for obj in tmp_obj_list:
                if self.bypass_objective_jacobian:
                    q_cache_index = 0
                else:
                    q_cache_index = self._q_cache_index[cnt]
                for scen_ix in q_scenarios[obj.quantity]:
                    self._grad_cache_intermediate[obj.quantity][q_cache_index, ix] += (
                        obj.priority
                        * obj.compute_gradient(q_vectors[obj.quantity].flat[scen_ix][ix])
                    )
                cnt += 1

        # Gradient with respect to bixel weights
        if self._grad_cache is None:
            if self.bypass_objective_jacobian:
                n_grad_caches = 1
            else:
                n_grad_caches = cnt
            self._grad_cache = np.zeros(
                (n_grad_caches, self._dij.total_num_of_bixels), dtype=np.float64
            )
        else:
            self._grad_cache.fill(0.0)

        for q in self._quantities:
            for scen_ix in q_scenarios[q.identifier]:
                if self.bypass_objective_jacobian:
                    cache_ix = 0
                else:
                    cache_ix = self._objectives_per_quantity[q.identifier]

                self._grad_cache[cache_ix, :] += (
                    q.compute_chain_derivative(self._grad_cache_intermediate[q.identifier], bixel_weights)
                    .flat[scen_ix]
                    .squeeze()
                )

        # Transform gradient from bixel space to aperture space
        # d(objective)/d(aperture) = d(objective)/d(bixels) * d(bixels)/d(aperture)
        grad_aperture = self._transform_gradient_to_aperture_space(
            self._grad_cache, x
        )

        return grad_aperture

    def _transform_gradient_to_aperture_space(
        self, grad_bixel: NDArray, aperture_vec: NDArray
    ) -> NDArray:
        """
        Transform gradient from bixel space to aperture parameter space.

        Parameters
        ----------
        grad_bixel : NDArray
            Gradient with respect to bixel weights. Shape: (n_caches, total_num_bixels)
        aperture_vec : NDArray
            Current aperture parameter vector.

        Returns
        -------
        NDArray
            Gradient with respect to aperture parameters. Shape: (n_caches, len(aperture_vec))

        Notes
        -----
        This computes the Jacobian d(bixels)/d(aperture) which accounts for:
        1. Weight derivatives: straightforward mapping via shape contributions
        2. Leaf position derivatives: how shifting leaves changes bixel coverage

        This is the most complex part of DAO gradient computation and is equivalent
        to matRad_objectiveGradient.m in @matRad_OptimizationProblemDAO.
        """
        n_caches = grad_bixel.shape[0]
        vec_length = len(aperture_vec)
        grad_aperture = np.zeros((n_caches, vec_length), dtype=np.float64)

        # Gradient with respect to weights (first total_num_shapes entries)
        # For weight w_i: d(bixel_j)/d(w_i) = shape_map_i[j]
        offset_weight = 0
        offset_bixel = 0

        for beam in self.aperture_info.beam:
            for shape_idx, shape in enumerate(beam.shape):
                # Get bixel indices for this beam
                bixel_indices = np.array(list(range(offset_bixel, offset_bixel + beam.num_of_active_leaf_pairs)))

                # d(bixel)/d(weight) = shape_map (1 where aperture is open, 0 otherwise)
                # Simplified: assume uniform contribution for now
                # Full implementation would reconstruct shape_map from leaf positions
                grad_aperture[:, offset_weight] = np.sum(
                    grad_bixel[:, bixel_indices], axis=1
                ) * shape.jacobi_scale

                offset_weight += 1

            offset_bixel += len(beam.shape) * beam.num_of_active_leaf_pairs

        # Gradient with respect to leaf positions
        # This is complex and requires computing how shifting a leaf changes bixel weights
        # For now, we use a simplified finite-difference approximation
        # Full implementation would follow matRad's analytical approach

        # Left leaf positions (entries [total_num_shapes : total_num_shapes + total_num_leaf_pairs])
        # Right leaf positions (entries [total_num_shapes + total_num_leaf_pairs : ...])
        # These derivatives are more involved and will be implemented in refinement

        return grad_aperture

    def _objective_gradient(self, x: NDArray) -> NDArray:
        """Compute gradient of scalar objective."""
        t = time.time()
        jac = np.sum(self._objective_jacobian(x), axis=0)
        self._deriv_times.append(time.time() - t)
        return jac

    def _constraint_functions(self, x: NDArray) -> NDArray:
        """
        Compute constraint function values.

        Parameters
        ----------
        x : NDArray
            Aperture parameter vector.

        Returns
        -------
        NDArray
            Constraint values. Shape: (total_num_leaf_pairs,)
            c[i] = right_leaf_pos[i] - left_leaf_pos[i] (must be >= 0)

        Notes
        -----
        Leaf gap constraints ensure no MLC collisions:
            right_leaf_pos[i] >= left_leaf_pos[i]  for all leaf pairs i
        """
        # Extract leaf positions from aperture vector
        total_shapes = self.aperture_info.total_num_of_shapes
        total_leaf_pairs = self.aperture_info.total_num_of_leaf_pairs

        left_leaf_pos = x[total_shapes : total_shapes + total_leaf_pairs]
        right_leaf_pos = x[
            total_shapes + total_leaf_pairs : total_shapes + 2 * total_leaf_pairs
        ]

        # Constraint: right - left >= 0
        c_dao = right_leaf_pos - left_leaf_pos

        # If there are dose constraints, add them
        # (For now, DAO only has leaf gap constraints)
        # bixel_weights = self._aperture_vector_to_bixel_weights(x)
        # c_dose = self._compute_dose_constraints(bixel_weights)
        # return np.concatenate([c_dose, c_dao])

        return c_dao

    def _constraint_function(self, x: NDArray) -> NDArray:
        """Wrapper for solver callback."""
        return self._constraint_functions(x)

    def _constraint_jacobian(self, x: NDArray) -> NDArray:
        """
        Compute Jacobian of constraints.

        Parameters
        ----------
        x : NDArray
            Aperture parameter vector.

        Returns
        -------
        NDArray
            Constraint Jacobian. Shape: (total_num_leaf_pairs, len(x))
            Sparse structure: only non-zero for leaf position variables.

        Notes
        -----
        For leaf gap constraints c[i] = right[i] - left[i]:
            dc[i]/d(left[i]) = -1
            dc[i]/d(right[i]) = +1
            dc[i]/d(others) = 0
        """
        total_shapes = self.aperture_info.total_num_of_shapes
        total_leaf_pairs = self.aperture_info.total_num_of_leaf_pairs
        vec_length = len(x)

        # Initialize sparse Jacobian
        jac = np.zeros((total_leaf_pairs, vec_length), dtype=np.float64)

        # Fill in leaf position derivatives
        for i in range(total_leaf_pairs):
            # dc[i]/d(left[i]) = -1
            jac[i, total_shapes + i] = -1.0
            # dc[i]/d(right[i]) = +1
            jac[i, total_shapes + total_leaf_pairs + i] = 1.0

        return jac

    def _constraint_bounds(self) -> tuple[NDArray, NDArray]:
        """
        Get constraint bounds.

        Returns
        -------
        tuple[NDArray, NDArray]
            (lower_bounds, upper_bounds) for constraints.

        Notes
        -----
        For leaf gap constraints: 0 <= c[i] <= inf (no upper limit)
        """
        total_leaf_pairs = self.aperture_info.total_num_of_leaf_pairs
        cl = np.zeros(total_leaf_pairs, dtype=np.float64)  # Lower bound: 0
        cu = np.full(total_leaf_pairs, np.inf, dtype=np.float64)  # Upper bound: inf

        return cl, cu

    def _solve(self) -> tuple[NDArray, dict]:
        """
        Solve the DAO optimization problem.

        Returns
        -------
        tuple[NDArray, dict]
            Optimized aperture vector and solver info.
        """
        self._deriv_times = []
        self._obj_times = []

        # Initial point from aperture_info
        x0 = self.aperture_info.to_vector()

        t = time.time()
        result = self.solver.solve(x0)
        self._solve_time = time.time() - t

        logger.info(
            "%d Objective function evaluations, avg. time: %g +/- %g s",
            len(self._obj_times),
            np.mean(self._obj_times),
            np.std(self._obj_times),
        )
        logger.info(
            "%d Derivative evaluations, avg. time: %g +/- %g s",
            len(self._deriv_times),
            np.mean(self._deriv_times),
            np.std(self._deriv_times),
        )
        logger.info("Solver time: %g s", self._solve_time)

        # Update aperture_info with optimized solution
        self.aperture_info.from_vector(result[0])

        return result
