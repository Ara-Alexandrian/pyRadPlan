"""Aperture data structures for Direct Aperture Optimization (DAO).

This module contains Pydantic models for representing MLC aperture information
used in Direct Aperture Optimization and VMAT optimization.

References
----------
matRad: matRad/matRad/sequencing/matRad_sequencing2ApertureInfo.m
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator

from pyRadPlan.core import PyRadPlanBaseModel


class ApertureShape(PyRadPlanBaseModel):
    """
    Represents a single MLC aperture shape.

    Attributes
    ----------
    left_leaf_pos : NDArray[np.float64]
        Left leaf positions in mm for each leaf pair. Shape: (num_leaf_pairs,)
    right_leaf_pos : NDArray[np.float64]
        Right leaf positions in mm for each leaf pair. Shape: (num_leaf_pairs,)
    weight : float
        Aperture weight (MU or fluence weight). Must be non-negative.
    shape_map : Optional[NDArray[np.bool_]]
        Boolean map indicating which bixels are open. Shape: (dimZ, dimX)
    jacobi_scale : float
        Scaling factor for Jacobian calculations. Default: 1.0
    vector_offset : Optional[int]
        Offset in the optimization vector for this shape's leaf positions.
    weight_offset : Optional[int]
        Offset in the optimization vector for this shape's weight.

    VMAT-specific attributes
    ------------------------
    mu_rate : Optional[float]
        Monitor unit delivery rate for VMAT (MU/min).
    left_leaf_pos_i : Optional[NDArray[np.float64]]
        Initial left leaf positions for continuous aperture VMAT.
    left_leaf_pos_f : Optional[NDArray[np.float64]]
        Final left leaf positions for continuous aperture VMAT.
    right_leaf_pos_i : Optional[NDArray[np.float64]]
        Initial right leaf positions for continuous aperture VMAT.
    right_leaf_pos_f : Optional[NDArray[np.float64]]
        Final right leaf positions for continuous aperture VMAT.

    Notes
    -----
    Leaf positions are in mm, measured from the isocenter in the BEV coordinate system.
    For each leaf pair, left_leaf_pos <= right_leaf_pos (no collisions).
    """

    left_leaf_pos: NDArray[np.float64] = Field(description="Left leaf positions in mm")
    right_leaf_pos: NDArray[np.float64] = Field(description="Right leaf positions in mm")
    weight: float = Field(ge=0.0, description="Aperture weight (MU)")
    shape_map: Optional[NDArray[np.bool_]] = Field(
        default=None, description="Boolean map of open bixels"
    )
    jacobi_scale: float = Field(default=1.0, description="Jacobian scaling factor")
    vector_offset: Optional[int] = Field(
        default=None, description="Offset in optimization vector"
    )
    weight_offset: Optional[int] = Field(
        default=None, description="Weight offset in optimization vector"
    )

    # VMAT-specific fields
    mu_rate: Optional[float] = Field(
        default=None, ge=0.0, description="MU delivery rate (VMAT)"
    )
    left_leaf_pos_i: Optional[NDArray[np.float64]] = Field(
        default=None, description="Initial left leaf positions (continuous VMAT)"
    )
    left_leaf_pos_f: Optional[NDArray[np.float64]] = Field(
        default=None, description="Final left leaf positions (continuous VMAT)"
    )
    right_leaf_pos_i: Optional[NDArray[np.float64]] = Field(
        default=None, description="Initial right leaf positions (continuous VMAT)"
    )
    right_leaf_pos_f: Optional[NDArray[np.float64]] = Field(
        default=None, description="Final right leaf positions (continuous VMAT)"
    )

    @field_validator("left_leaf_pos", "right_leaf_pos", mode="before")
    @classmethod
    def validate_leaf_positions(cls, v):
        """Ensure leaf positions are numpy arrays."""
        if not isinstance(v, np.ndarray):
            return np.array(v, dtype=np.float64)
        return v.astype(np.float64)

    @field_validator("shape_map", mode="before")
    @classmethod
    def validate_shape_map(cls, v):
        """Ensure shape map is a boolean numpy array or None."""
        if v is None:
            return None
        if not isinstance(v, np.ndarray):
            return np.array(v, dtype=bool)
        return v.astype(bool)


class ApertureBeam(PyRadPlanBaseModel):
    """
    Represents aperture information for a single beam angle.

    Attributes
    ----------
    shape : List[ApertureShape]
        List of aperture shapes for this beam.
    num_of_shapes : int
        Number of shapes (segments) in this beam. Must match len(shape).
    num_of_active_leaf_pairs : int
        Number of active MLC leaf pairs for this beam.
    leaf_pair_pos : NDArray[np.float64]
        Z-coordinates of active leaf pairs in mm. Shape: (num_of_active_leaf_pairs,)
    is_active_leaf_pair : NDArray[np.bool_]
        Boolean array indicating which leaf pairs are active (1=active, 0=inactive).
        Shape: (num_of_mlc_leaf_pairs,), typically 80.
    central_leaf_pair : int
        Index of the central leaf pair (e.g., 40 for 80 leaf pairs).
    lim_l : NDArray[np.float64]
        Left limit (minimum position) for each leaf pair. Shape: (num_of_active_leaf_pairs,)
    lim_r : NDArray[np.float64]
        Right limit (maximum position) for each leaf pair. Shape: (num_of_active_leaf_pairs,)
    bixel_ind_map : NDArray[np.float64]
        Map of bixel indices. Shape: (dimZ, dimX). NaN where no bixel exists.
    pos_of_corner_bixel : NDArray[np.float64]
        Physical position [x, y, z] of the corner bixel (top-left). Shape: (3,)
    mlc_window : NDArray[np.float64]
        MLC window bounds [min_x, max_x, min_z, max_z] in mm. Shape: (4,)
    gantry_angle : float
        Gantry angle for this beam in degrees.
    bix_offset : Optional[int]
        Bixel offset for gradient calculations (VMAT only).
    gantry_rot : Optional[float]
        Gantry rotation speed in deg/sec (VMAT only).

    Notes
    -----
    - Z-direction is perpendicular to leaf motion (across leaf pairs)
    - X-direction is parallel to leaf motion (leaf travel direction)
    - BEV = Beam's Eye View coordinate system
    """

    shape: List[ApertureShape] = Field(
        default_factory=list, description="List of aperture shapes"
    )
    num_of_shapes: int = Field(ge=0, description="Number of shapes for this beam")
    num_of_active_leaf_pairs: int = Field(
        ge=1, description="Number of active leaf pairs"
    )
    leaf_pair_pos: NDArray[np.float64] = Field(
        description="Z-coordinates of leaf pairs"
    )
    is_active_leaf_pair: NDArray[np.bool_] = Field(
        description="Boolean mask of active leaf pairs"
    )
    central_leaf_pair: int = Field(
        ge=1, description="Index of central leaf pair"
    )
    lim_l: NDArray[np.float64] = Field(
        description="Left limits for each leaf pair"
    )
    lim_r: NDArray[np.float64] = Field(
        description="Right limits for each leaf pair"
    )
    bixel_ind_map: NDArray[np.float64] = Field(
        description="Map of bixel indices"
    )
    pos_of_corner_bixel: NDArray[np.float64] = Field(
        description="Physical position of corner bixel [x, y, z]"
    )
    mlc_window: NDArray[np.float64] = Field(
        description="MLC window bounds [min_x, max_x, min_z, max_z]"
    )
    gantry_angle: float = Field(description="Gantry angle in degrees")

    # VMAT-specific fields
    bix_offset: Optional[int] = Field(
        default=None, description="Bixel offset for gradient calc (VMAT)"
    )
    gantry_rot: Optional[float] = Field(
        default=None, description="Gantry rotation speed (deg/sec, VMAT)"
    )

    @field_validator("num_of_shapes", mode="after")
    @classmethod
    def validate_num_shapes(cls, v, info):
        """Ensure num_of_shapes matches the length of shape list."""
        if "shape" in info.data and len(info.data["shape"]) != v:
            raise ValueError(
                f"num_of_shapes ({v}) must match length of shape list ({len(info.data['shape'])})"
            )
        return v


class VMATProperties(PyRadPlanBaseModel):
    """
    VMAT-specific properties for aperture optimization.

    Attributes
    ----------
    constraints : Dict[str, Any]
        Machine constraints for VMAT delivery (leaf speed, gantry speed, etc.)
    jacobi_t : NDArray[np.float64]
        Jacobian transformation matrix for time-based interpolation.
        Shape: (total_num_shapes, num_beams)
    num_leaf_speed_constraint : int
        Number of leaf speed constraints (transitions between beams).
    num_leaf_speed_constraint_dao : int
        Number of DAO leaf speed constraints.
    machine_constraint_file : Optional[str]
        Path to machine constraint file.
    beam : List[Dict[str, Any]]
        Beam-specific VMAT properties (DAO flags, angle borders, time factors, etc.)
    """

    constraints: Dict[str, Any] = Field(
        default_factory=dict, description="VMAT machine constraints"
    )
    jacobi_t: NDArray[np.float64] = Field(
        description="Jacobian transformation matrix"
    )
    num_leaf_speed_constraint: int = Field(
        default=0, ge=0, description="Number of leaf speed constraints"
    )
    num_leaf_speed_constraint_dao: int = Field(
        default=0, ge=0, description="Number of DAO leaf speed constraints"
    )
    machine_constraint_file: Optional[str] = Field(
        default=None, description="Path to machine constraint file"
    )
    beam: List[Dict[str, Any]] = Field(
        default_factory=list, description="Beam-specific VMAT properties"
    )


class ApertureInfo(PyRadPlanBaseModel):
    """
    Complete aperture information for Direct Aperture Optimization.

    This is the main data structure for DAO and VMAT optimization, containing
    all MLC aperture shapes, weights, limits, and optimization metadata.

    Attributes
    ----------
    beam : List[ApertureBeam]
        List of aperture beams, one per gantry angle.
    bixel_width : float
        Width of bixels in mm (typically 5mm).
    num_of_mlc_leaf_pairs : int
        Total number of MLC leaf pairs (typically 80).
    total_num_of_bixels : int
        Total number of bixels across all beams.
    total_num_of_shapes : int
        Total number of aperture shapes across all beams.
    total_num_of_opt_bixels : Optional[int]
        Total number of optimization bixels (VMAT-specific).
    total_num_of_leaf_pairs : int
        Total number of leaf pair variables (shapes × leaf_pairs).
    dose_total_num_of_leaf_pairs : int
        Total number of leaf pairs for dose calculation.
    aperture_vector : Optional[NDArray[np.float64]]
        Flattened optimization vector [weights, left_positions, right_positions, times].
        Shape: (total_num_shapes + 2*total_num_leaf_pairs [+ total_num_shapes for VMAT],)
    mapping_mx : Optional[NDArray[np.float64]]
        Mapping matrix for optimization vector to beam/shape/leaf indices.
        Shape: (len(aperture_vector), 4). Columns: [beam_idx, global_shape_idx, local_shape_idx, leaf_idx]
    lim_mx : Optional[NDArray[np.float64]]
        Bounds matrix for optimization vector. Shape: (len(aperture_vector), 2).
        Columns: [lower_bound, upper_bound]
    jacobi_scale : NDArray[np.float64]
        Scaling factors for Jacobian. Shape: (total_num_shapes,)
    weight_to_mu : Optional[float]
        Conversion factor from optimization weights to Monitor Units.
    continuous_aperture : bool
        Flag for continuous aperture VMAT optimization. Default: False.
    run_vmat : bool
        Flag indicating VMAT optimization mode. Default: False.
    preconditioner : bool
        Flag for using preconditioning in optimization. Default: False.
    prop_vmat : Optional[VMATProperties]
        VMAT-specific properties (only present if run_vmat=True).

    Methods
    -------
    to_vector() -> NDArray[np.float64]
        Convert aperture info to optimization vector.
    from_vector(vec: NDArray[np.float64]) -> None
        Update aperture info from optimization vector.
    get_bounds() -> Tuple[NDArray[np.float64], NDArray[np.float64]]
        Get lower and upper bounds for optimization.

    Notes
    -----
    Vector structure:
        - [0 : total_num_shapes] = aperture weights
        - [total_num_shapes : total_num_shapes + total_num_leaf_pairs] = left leaf positions
        - [... + total_num_leaf_pairs : ... + 2*total_num_leaf_pairs] = right leaf positions
        - [... (VMAT only) : end] = arc sector times

    References
    ----------
    matRad: matRad/matRad/sequencing/matRad_sequencing2ApertureInfo.m
    matRad: matRad/matRad/optimization/@matRad_OptimizationProblemDAO/matRad_daoApertureInfo2Vec.m
    """

    beam: List[ApertureBeam] = Field(
        default_factory=list, description="List of aperture beams"
    )
    bixel_width: float = Field(gt=0.0, description="Bixel width in mm")
    num_of_mlc_leaf_pairs: int = Field(
        default=80, ge=1, description="Total number of MLC leaf pairs"
    )
    total_num_of_bixels: int = Field(
        ge=0, description="Total number of bixels"
    )
    total_num_of_shapes: int = Field(
        ge=0, description="Total number of shapes"
    )
    total_num_of_opt_bixels: Optional[int] = Field(
        default=None, description="Total number of optimization bixels (VMAT)"
    )
    total_num_of_leaf_pairs: int = Field(
        ge=0, description="Total number of leaf pair variables"
    )
    dose_total_num_of_leaf_pairs: int = Field(
        ge=0, description="Total leaf pairs for dose calculation"
    )
    aperture_vector: Optional[NDArray[np.float64]] = Field(
        default=None, description="Flattened optimization vector"
    )
    mapping_mx: Optional[NDArray[np.float64]] = Field(
        default=None, description="Mapping matrix for optimization"
    )
    lim_mx: Optional[NDArray[np.float64]] = Field(
        default=None, description="Bounds matrix"
    )
    jacobi_scale: NDArray[np.float64] = Field(
        description="Jacobian scaling factors"
    )
    weight_to_mu: Optional[float] = Field(
        default=None, description="Weight to MU conversion factor"
    )
    continuous_aperture: bool = Field(
        default=False, description="Continuous aperture VMAT flag"
    )
    run_vmat: bool = Field(
        default=False, description="VMAT optimization flag"
    )
    preconditioner: bool = Field(
        default=False, description="Preconditioning flag"
    )
    prop_vmat: Optional[VMATProperties] = Field(
        default=None, description="VMAT-specific properties"
    )

    def to_vector(self) -> NDArray[np.float64]:
        """
        Convert aperture info to optimization vector.

        Returns
        -------
        NDArray[np.float64]
            Flattened optimization vector containing weights, leaf positions,
            and (for VMAT) arc sector times.

        Notes
        -----
        This is equivalent to matRad_daoApertureInfo2Vec.m.
        Vector structure: [weights, left_positions, right_positions, times (VMAT only)]
        """
        vec_length = (
            self.total_num_of_shapes + self.total_num_of_leaf_pairs * 2
        )
        if self.run_vmat:
            vec_length += self.total_num_of_shapes

        vec = np.full(vec_length, np.nan, dtype=np.float64)
        offset = 0

        # 1. Fill aperture weights
        for beam in self.beam:
            for shape in beam.shape:
                vec[offset] = shape.jacobi_scale * shape.weight
                offset += 1

        # 2. Fill left and right leaf positions
        offset_left = self.total_num_of_shapes
        offset_right = self.total_num_of_shapes + self.total_num_of_leaf_pairs

        for beam in self.beam:
            for shape in beam.shape:
                n_leaves = beam.num_of_active_leaf_pairs
                vec[offset_left:offset_left + n_leaves] = shape.left_leaf_pos
                vec[offset_right:offset_right + n_leaves] = shape.right_leaf_pos
                offset_left += n_leaves
                offset_right += n_leaves

        # 3. Fill VMAT arc sector times (if applicable)
        if self.run_vmat and self.prop_vmat is not None:
            offset = self.total_num_of_shapes + 2 * self.total_num_of_leaf_pairs
            # Implementation depends on VMAT arc sequencing
            # This will be completed when VMAT is implemented
            pass

        return vec

    def from_vector(self, vec: NDArray[np.float64]) -> None:
        """
        Update aperture info from optimization vector.

        Parameters
        ----------
        vec : NDArray[np.float64]
            Optimization vector from DAO/VMAT optimizer.

        Notes
        -----
        This is the inverse of to_vector(), equivalent to matRad_daoVec2ApertureInfo.m.
        Updates weights and leaf positions in-place.
        """
        offset = 0

        # 1. Extract aperture weights
        for beam in self.beam:
            for shape in beam.shape:
                shape.weight = vec[offset] / shape.jacobi_scale
                offset += 1

        # 2. Extract left and right leaf positions
        offset_left = self.total_num_of_shapes
        offset_right = self.total_num_of_shapes + self.total_num_of_leaf_pairs

        for beam in self.beam:
            for shape in beam.shape:
                n_leaves = beam.num_of_active_leaf_pairs
                shape.left_leaf_pos = vec[offset_left:offset_left + n_leaves].copy()
                shape.right_leaf_pos = vec[offset_right:offset_right + n_leaves].copy()
                offset_left += n_leaves
                offset_right += n_leaves

        # 3. Extract VMAT arc sector times (if applicable)
        if self.run_vmat and self.prop_vmat is not None:
            # Implementation depends on VMAT arc sequencing
            pass

    def get_bounds(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get lower and upper bounds for optimization.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            (lower_bounds, upper_bounds) arrays for the optimization vector.

        Notes
        -----
        - Weights: [0, inf]
        - Leaf positions: [lim_l, lim_r] for each leaf pair
        - VMAT times: [min_time, max_time] based on gantry speed constraints
        """
        if self.lim_mx is not None:
            return self.lim_mx[:, 0], self.lim_mx[:, 1]
        else:
            raise ValueError("lim_mx not initialized. Call aperture_info_to_vec first.")


def sequencing_to_aperture_info(
    sequencing: Dict[str, Any],
    stf: List[Dict[str, Any]],
    pln: Any,
) -> ApertureInfo:
    """
    Convert sequencing results to ApertureInfo structure.

    Parameters
    ----------
    sequencing : Dict[str, Any]
        Sequencing results from Siochi/Engel/Xia algorithm.
        Contains: beam (list), w (weights), weightToMU (conversion factor)
    stf : List[Dict[str, Any]]
        Steering information with beam geometry and bixel positions.
    pln : Any
        Treatment plan object with optimization and sequencing properties.

    Returns
    -------
    ApertureInfo
        Complete aperture information ready for DAO optimization.

    Notes
    -----
    This is the Python equivalent of matRad_sequencing2ApertureInfo.m.
    Converts discrete sequencing output (shapes with binary maps) into
    continuous aperture representation (leaf positions with limits).

    References
    ----------
    matRad: matRad/matRad/sequencing/matRad_sequencing2ApertureInfo.m
    """
    # Implementation will be completed in next phase
    # This requires reading STF structure and converting shape maps to leaf positions
    raise NotImplementedError("sequencing_to_aperture_info will be implemented in Phase 2")
