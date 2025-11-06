"""Volumetric Modulated Arc Therapy (VMAT) optimization for pyRadPlan.

Extends Direct Aperture Optimization with arc-specific constraints including
leaf speed limits and dose rate limits for deliverable VMAT plans.

References
----------
matRad: matRad/matRad/optimization/@matRad_OptimizationProblemVMAT/
"""

from typing import Union
import logging

import numpy as np
from numpy.typing import NDArray

from ...plan import Plan
from ...sequencing import ApertureInfo
from ._dao import DirectApertureOptimization

logger = logging.getLogger(__name__)


class VMATOptimization(DirectApertureOptimization):
    """
    Volumetric Modulated Arc Therapy (VMAT) optimization problem.

    Extends Direct Aperture Optimization with additional constraints for
    continuous arc delivery:
    1. Leaf speed constraints: limit how fast MLC leaves can move
    2. Dose rate constraints: limit monitor unit delivery rate

    Parameters
    ----------
    pln : Union[Plan, dict], optional
        Plan object or dictionary to initialize the problem with.
    aperture_info : ApertureInfo, optional
        Initial aperture information from VMAT arc sequencing.

    Attributes
    ----------
    aperture_info : ApertureInfo
        Aperture information with VMAT-specific properties.
        Must have aperture_info.run_vmat = True.

    Notes
    -----
    VMAT optimization variables extend DAO:
        x = [weights, left_leaf_positions, right_leaf_positions, arc_times]
    where:
        - arc_times: time spent in each arc sector (total_num_shapes,)

    VMAT constraints extend DAO leaf gaps with:
        1. Leaf speed: |pos_final - pos_initial| / time <= max_leaf_speed
        2. Dose rate: weight / time within [min_MU_rate, max_MU_rate]

    For continuous aperture mode:
        - Leaf positions are defined at arc sector borders (initial and final)
        - Speed constraints apply to transitions between sectors
        - More complex indexing for interpolated beams

    References
    ----------
    matRad: matRad/matRad/optimization/@matRad_OptimizationProblemVMAT/matRad_OptimizationProblemVMAT.m
    """

    name = "Volumetric Modulated Arc Therapy Optimization"
    short_name = "vmat"

    def __init__(
        self,
        pln: Union[Plan, dict] = None,
        aperture_info: ApertureInfo = None,
    ):
        super().__init__(pln, aperture_info)

    def _initialize(self):
        """Initialize the VMAT problem."""
        super()._initialize()

        # Validate VMAT-specific requirements
        if not self.aperture_info.run_vmat:
            raise ValueError(
                "aperture_info.run_vmat must be True for VMAT optimization. "
                "Use DirectApertureOptimization for non-VMAT cases."
            )

        if self.aperture_info.prop_vmat is None:
            raise ValueError(
                "aperture_info.prop_vmat must be set for VMAT optimization. "
                "Ensure VMAT arc sequencing was performed."
            )

        # Update constraint bounds to include VMAT constraints
        self.solver.constraint_bounds = self._constraint_bounds()

    def _constraint_functions(self, x: NDArray) -> NDArray:
        """
        Compute constraint function values for VMAT.

        Parameters
        ----------
        x : NDArray
            Aperture parameter vector including arc times.

        Returns
        -------
        NDArray
            Constraint values: [dose_constraints, leaf_gaps, leaf_speeds, dose_rates]

        Notes
        -----
        VMAT adds two new constraint types beyond DAO:
        1. Leaf speed constraints: v = |Δpos| / Δt <= v_max
        2. Dose rate constraints: MU_rate = weight / time within bounds
        """
        # Get DAO constraints (dose + leaf gaps)
        c_dao = super()._constraint_functions(x)

        # Extract components from aperture vector
        total_shapes = self.aperture_info.total_num_of_shapes
        total_leaf_pairs = self.aperture_info.total_num_of_leaf_pairs

        left_leaf_pos = x[total_shapes : total_shapes + total_leaf_pairs]
        right_leaf_pos = x[
            total_shapes + total_leaf_pairs : total_shapes + 2 * total_leaf_pairs
        ]
        weights = x[:total_shapes] / self.aperture_info.jacobi_scale
        arc_times = x[total_shapes + 2 * total_leaf_pairs :]

        # Compute arc sector times accounting for time factors
        # This maps DAO border times to dose influence border times
        time_dao_border_angles = arc_times
        dao_beams = [beam for beam in self.aperture_info.prop_vmat.beam if beam.get('DAOBeam', False)]
        time_fac_curr = np.array([beam['timeFacCurr'] for beam in dao_beams])
        time_dose_border_angles = time_dao_border_angles * time_fac_curr

        # Leaf speed constraints
        c_leaf_speed = self._compute_leaf_speed_constraints(
            left_leaf_pos, right_leaf_pos, time_dao_border_angles
        )

        # Dose rate constraints
        c_dose_rate = self._compute_dose_rate_constraints(
            weights, time_dose_border_angles
        )

        # Concatenate all constraints
        return np.concatenate([c_dao, c_leaf_speed, c_dose_rate])

    def _compute_leaf_speed_constraints(
        self,
        left_leaf_pos: NDArray,
        right_leaf_pos: NDArray,
        arc_times: NDArray,
    ) -> NDArray:
        """
        Compute leaf speed constraints for VMAT.

        Parameters
        ----------
        left_leaf_pos : NDArray
            All left leaf positions.
        right_leaf_pos : NDArray
            All right leaf positions.
        arc_times : NDArray
            Time for each arc sector.

        Returns
        -------
        NDArray
            Leaf speed values. Shape: (num_speed_constraints * num_leaf_pairs * 2,)
            Constraint: speed <= max_leaf_speed

        Notes
        -----
        For continuous aperture:
            - Each DAO beam has initial and final leaf positions
            - Speed = |pos_final - pos_initial| / time_between_sectors

        For discrete aperture:
            - Speed = |pos[i+1] - pos[i]| / time[i]
            - Applied between consecutive control points
        """
        if self.aperture_info.continuous_aperture:
            return self._compute_continuous_leaf_speeds(
                left_leaf_pos, right_leaf_pos, arc_times
            )
        else:
            return self._compute_discrete_leaf_speeds(
                left_leaf_pos, right_leaf_pos, arc_times
            )

    def _compute_continuous_leaf_speeds(
        self,
        left_leaf_pos: NDArray,
        right_leaf_pos: NDArray,
        arc_times: NDArray,
    ) -> NDArray:
        """
        Compute leaf speeds for continuous aperture VMAT.

        Parameters
        ----------
        left_leaf_pos, right_leaf_pos : NDArray
            Leaf positions (includes both initial and final for each DAO beam).
        arc_times : NDArray
            Time for each arc sector.

        Returns
        -------
        NDArray
            Leaf speed constraints.

        Notes
        -----
        Uses time factors to compute actual transition times between
        control points, accounting for interpolated beams.
        """
        num_constraints = self.aperture_info.prop_vmat.num_leaf_speed_constraint
        num_leaf_pairs = self.aperture_info.beam[0].num_of_active_leaf_pairs

        left_speeds = np.zeros(num_constraints * num_leaf_pairs, dtype=np.float64)
        right_speeds = np.zeros(num_constraints * num_leaf_pairs, dtype=np.float64)

        # Build time factor matrix for arc sector times
        time_fac_list = []
        time_fac_ind_list = []
        for beam in self.aperture_info.prop_vmat.beam:
            if 'timeFac' in beam and 'timeFacInd' in beam:
                time_fac_list.extend(beam['timeFac'])
                time_fac_ind_list.extend(beam['timeFacInd'])

        # Remove zero entries
        time_fac = np.array(time_fac_list)
        time_fac_ind = np.array(time_fac_ind_list)
        non_zero_mask = time_fac != 0
        time_fac = time_fac[non_zero_mask]
        time_fac_ind = time_fac_ind[non_zero_mask]

        # Create sparse mapping (simplified - full implementation would use scipy.sparse)
        # time_bn_opt_angles = time_factor_matrix @ arc_times
        # For now, use direct calculation
        time_bn_opt_angles = arc_times  # Simplified

        offset = 0
        shape_idx = 0

        for beam_idx, beam in enumerate(self.aperture_info.beam):
            n = beam.num_of_active_leaf_pairs

            if self.aperture_info.prop_vmat.beam[beam_idx].get('leafConstMask'):
                # Get vector indices for initial and final positions
                if self.aperture_info.prop_vmat.beam[beam_idx].get('DAOBeam', False):
                    # DAO beam: use own vector offset
                    vector_offset = beam.shape[0].vector_offset
                    if isinstance(vector_offset, tuple) or isinstance(vector_offset, list):
                        vector_ix_li = np.arange(vector_offset[0], vector_offset[0] + n)
                        vector_ix_lf = np.arange(vector_offset[1], vector_offset[1] + n)
                    else:
                        vector_ix_li = np.arange(vector_offset, vector_offset + n)
                        vector_ix_lf = vector_ix_li  # Same positions (no continuous)
                else:
                    # Interpolated beam: use previous and next DAO indices
                    last_dao_idx = self.aperture_info.prop_vmat.beam[beam_idx]['lastDAOIndex']
                    next_dao_idx = self.aperture_info.prop_vmat.beam[beam_idx]['nextDAOIndex']
                    last_offset = self.aperture_info.beam[last_dao_idx].shape[0].vector_offset
                    next_offset = self.aperture_info.beam[next_dao_idx].shape[0].vector_offset

                    if isinstance(last_offset, tuple):
                        vector_ix_li = np.arange(last_offset[1], last_offset[1] + n)
                    else:
                        vector_ix_li = np.arange(last_offset, last_offset + n)

                    if isinstance(next_offset, tuple):
                        vector_ix_lf = np.arange(next_offset[0], next_offset[0] + n)
                    else:
                        vector_ix_lf = np.arange(next_offset, next_offset + n)

                # Compute indices in full vector (accounting for total_shapes offset)
                total_shapes = self.aperture_info.total_num_of_shapes
                total_leaf_pairs = self.aperture_info.total_num_of_leaf_pairs

                # Extract positions and time
                left_pos_i = left_leaf_pos[vector_ix_li - total_shapes]
                left_pos_f = left_leaf_pos[vector_ix_lf - total_shapes]
                right_pos_i = right_leaf_pos[vector_ix_li - total_shapes]
                right_pos_f = right_leaf_pos[vector_ix_lf - total_shapes]

                t = time_bn_opt_angles[shape_idx] if shape_idx < len(time_bn_opt_angles) else 1.0

                # Compute speeds
                ind_in_con_vec = np.arange(offset, offset + n)
                left_speeds[ind_in_con_vec] = np.abs(left_pos_f - left_pos_i) / t
                right_speeds[ind_in_con_vec] = np.abs(right_pos_f - right_pos_i) / t

                offset += n
                shape_idx += 1

        return np.concatenate([left_speeds, right_speeds])

    def _compute_discrete_leaf_speeds(
        self,
        left_leaf_pos: NDArray,
        right_leaf_pos: NDArray,
        arc_times: NDArray,
    ) -> NDArray:
        """
        Compute leaf speeds for discrete aperture VMAT.

        Parameters
        ----------
        left_leaf_pos, right_leaf_pos : NDArray
            Leaf positions for all shapes.
        arc_times : NDArray
            Time for each arc sector.

        Returns
        -------
        NDArray
            Leaf speed constraints between consecutive control points.
        """
        num_leaf_pairs = self.aperture_info.beam[0].num_of_active_leaf_pairs
        num_shapes = self.aperture_info.total_num_of_shapes

        # Reshape leaf positions: (num_leaf_pairs, num_shapes)
        left_pos_matrix = left_leaf_pos.reshape(num_leaf_pairs, num_shapes)
        right_pos_matrix = right_leaf_pos.reshape(num_leaf_pairs, num_shapes)

        # Compute time factors between consecutive shapes
        dao_beams = [beam for beam in self.aperture_info.prop_vmat.beam if beam.get('DAOBeam', False)]
        time_fac = []
        for beam in dao_beams:
            if 'timeFac' in beam:
                time_fac.extend(beam['timeFac'])

        time_fac = np.array(time_fac)
        # Remove first and last (no transitions there)
        time_fac = time_fac[1:-1]

        # Build sparse time factor matrix (simplified)
        # time_bn_opt_angles = time_factor_matrix @ arc_times
        time_bn_opt_angles = arc_times[:-1]  # Simplified

        # Compute position differences
        left_diff = np.abs(np.diff(left_pos_matrix, axis=1))  # (num_leaf_pairs, num_shapes-1)
        right_diff = np.abs(np.diff(right_pos_matrix, axis=1))

        # Compute speeds
        left_speeds = (left_diff / time_bn_opt_angles).flatten()
        right_speeds = (right_diff / time_bn_opt_angles).flatten()

        return np.concatenate([left_speeds, right_speeds])

    def _compute_dose_rate_constraints(
        self, weights: NDArray, time_dose_angles: NDArray
    ) -> NDArray:
        """
        Compute dose rate (MU/sec) constraints.

        Parameters
        ----------
        weights : NDArray
            Aperture weights for each shape.
        time_dose_angles : NDArray
            Time for dose delivery at each angle.

        Returns
        -------
        NDArray
            Dose rates (MU/sec) for each optimized angle.

        Notes
        -----
        Dose rate = (weight * weight_to_MU) / time
        Must be within machine limits: [min_MU_rate, max_MU_rate]
        """
        weight_to_mu = self.aperture_info.weight_to_mu
        if weight_to_mu is None:
            weight_to_mu = 1.0  # Default if not specified

        dose_rates = (weight_to_mu * weights) / time_dose_angles
        return dose_rates

    def _constraint_jacobian(self, x: NDArray) -> NDArray:
        """
        Compute Jacobian of VMAT constraints.

        Parameters
        ----------
        x : NDArray
            Aperture parameter vector.

        Returns
        -------
        NDArray
            Constraint Jacobian including VMAT-specific constraints.

        Notes
        -----
        Extends DAO Jacobian with derivatives of:
        1. Leaf speed w.r.t. leaf positions and arc times
        2. Dose rate w.r.t. weights and arc times
        """
        # Get DAO Jacobian (dose + leaf gaps)
        jac_dao = super()._constraint_jacobian(x)

        # Compute VMAT constraint derivatives
        # (Simplified - full implementation would compute analytical derivatives)
        # For now, return DAO Jacobian extended with zeros for VMAT constraints
        total_shapes = self.aperture_info.total_num_of_shapes
        total_leaf_pairs = self.aperture_info.total_num_of_leaf_pairs
        vec_length = len(x)

        # Count VMAT constraints
        if self.aperture_info.continuous_aperture:
            num_speed_constraints = (
                2 * self.aperture_info.prop_vmat.num_leaf_speed_constraint
                * self.aperture_info.beam[0].num_of_active_leaf_pairs
            )
        else:
            dao_count = sum(1 for b in self.aperture_info.prop_vmat.beam if b.get('DAOBeam', False))
            num_speed_constraints = (
                2 * (dao_count - 1) * self.aperture_info.beam[0].num_of_active_leaf_pairs
            )

        num_dose_rate_constraints = sum(
            1 for b in self.aperture_info.prop_vmat.beam if b.get('DAOBeam', False)
        )

        num_vmat_constraints = num_speed_constraints + num_dose_rate_constraints

        # Initialize VMAT Jacobian rows
        jac_vmat = np.zeros((num_vmat_constraints, vec_length), dtype=np.float64)

        # TODO: Fill in analytical derivatives for leaf speed and dose rate
        # This requires careful indexing based on continuous vs discrete mode

        # Concatenate Jacobians
        jac = np.vstack([jac_dao, jac_vmat])
        return jac

    def _constraint_bounds(self) -> tuple[NDArray, NDArray]:
        """
        Get constraint bounds for VMAT.

        Returns
        -------
        tuple[NDArray, NDArray]
            (lower_bounds, upper_bounds) for all constraints.

        Notes
        -----
        Extends DAO bounds with:
        - Leaf speed: [min_speed, max_speed] (typically [0, 25 mm/s])
        - Dose rate: [min_MU_rate, max_MU_rate] (typically [0, 600 MU/min])
        """
        # Get DAO constraint bounds (dose + leaf gaps)
        cl_dao, cu_dao = super()._constraint_bounds()

        # Leaf speed bounds
        constraints = self.aperture_info.prop_vmat.constraints
        num_leaf_pairs = self.aperture_info.beam[0].num_of_active_leaf_pairs

        if self.aperture_info.continuous_aperture:
            num_speed_constraints = (
                self.aperture_info.prop_vmat.num_leaf_speed_constraint
            )
        else:
            dao_count = sum(1 for b in self.aperture_info.prop_vmat.beam if b.get('DAOBeam', False))
            num_speed_constraints = dao_count - 1

        total_speed_constraints = 2 * num_speed_constraints * num_leaf_pairs

        min_leaf_speed = constraints.get('leafSpeed', [0.0, 25.0])[0]
        max_leaf_speed = constraints.get('leafSpeed', [0.0, 25.0])[1]

        cl_leaf_speed = np.full(total_speed_constraints, min_leaf_speed, dtype=np.float64)
        cu_leaf_speed = np.full(total_speed_constraints, max_leaf_speed, dtype=np.float64)

        # Dose rate bounds
        num_dose_rate = sum(1 for b in self.aperture_info.prop_vmat.beam if b.get('DAOBeam', False))

        min_mu_rate = constraints.get('monitorUnitRate', [0.0, 600.0])[0]
        max_mu_rate = constraints.get('monitorUnitRate', [0.0, 600.0])[1]

        cl_dose_rate = np.full(num_dose_rate, min_mu_rate, dtype=np.float64)
        cu_dose_rate = np.full(num_dose_rate, max_mu_rate, dtype=np.float64)

        # Concatenate all bounds
        cl = np.concatenate([cl_dao, cl_leaf_speed, cl_dose_rate])
        cu = np.concatenate([cu_dao, cu_leaf_speed, cu_dose_rate])

        return cl, cu
