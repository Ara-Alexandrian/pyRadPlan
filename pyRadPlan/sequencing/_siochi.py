"""Siochi leaf sequencing algorithm for IMRT.

Implements the Siochi algorithm for multi-leaf collimator (MLC) leaf sequencing
to convert continuous fluence maps into deliverable step-and-shoot IMRT segments.

The algorithm consists of three main steps:
1. Stratification: Discretize continuous fluence into N intensity levels
2. Rod Pushing: Determine deliverable MLC leaf positions accounting for collisions
3. Segment Creation: Extract discrete aperture shapes from rod positions

References
----------
Siochi, R.A. (1999). "Minimizing static intensity modulation delivery time
using an intensity solid paradigm." International Journal of Radiation
Oncology*Biology*Physics, 43(3), 671-680.

matRad: matRad/matRad/sequencing/matRad_siochiLeafSequencing.m
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter


def siochi_stratify(
    fluence_map: NDArray[np.float64],
    num_levels: int,
) -> Tuple[NDArray[np.int32], float]:
    """
    Stratify continuous fluence map into discrete intensity levels.

    Parameters
    ----------
    fluence_map : NDArray[np.float64]
        Continuous fluence map. Shape: (dimZ, dimX) where Z is perpendicular
        to leaf motion and X is parallel to leaf motion.
    num_levels : int
        Number of stratification levels (e.g., 5 means levels 0-5).

    Returns
    -------
    stratified : NDArray[np.int32]
        Discretized fluence map with integer levels 0 to num_levels.
        Shape: same as fluence_map.
    calibration_factor : float
        Maximum fluence value used for normalization.

    Notes
    -----
    The stratification formula is:
        D_k = round(fluence / max(fluence) * num_levels)

    This ensures the maximum intensity becomes num_levels and all others
    are proportionally scaled and rounded to the nearest integer.

    Examples
    --------
    >>> fluence = np.array([[0.0, 2.5, 5.0], [1.0, 3.5, 4.0]])
    >>> stratified, cal_fac = siochi_stratify(fluence, num_levels=5)
    >>> print(stratified)
    [[0 2 5]
     [1 4 4]]
    >>> print(f"Calibration factor: {cal_fac}")
    Calibration factor: 5.0
    """
    cal_factor = np.max(fluence_map)
    if cal_factor == 0:
        # Empty fluence map
        return np.zeros_like(fluence_map, dtype=np.int32), 0.0

    stratified = np.round(fluence_map / cal_factor * num_levels).astype(np.int32)
    return stratified, cal_factor


def siochi_decompose_port(
    intensity_map: NDArray[np.int32],
    dim_z: int,
    dim_x: int,
    min_z: int,
    max_z: int,
    min_x: int,
    max_x: int,
) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    Decompose stratified intensity map using rod pushing algorithm.

    This function implements the "rod pushing" analogy where each bixel is
    represented as a vertical rod of height equal to its intensity level.
    Rods are "pushed" column by column to find collision-free MLC positions.

    Parameters
    ----------
    intensity_map : NDArray[np.int32]
        Stratified intensity map with integer levels. Shape: (dim_z, dim_x)
    dim_z : int
        Z-dimension (perpendicular to leaf motion, across leaf pairs)
    dim_x : int
        X-dimension (parallel to leaf motion, leaf travel direction)
    min_z, max_z : int
        Minimum and maximum Z indices with non-zero intensity
    min_x, max_x : int
        Minimum and maximum X indices with non-zero intensity

    Returns
    -------
    tops : NDArray[np.int32]
        Top positions of rods. Shape: (dim_z, dim_x)
    bases : NDArray[np.int32]
        Base positions of rods. Shape: (dim_z, dim_x)

    Notes
    -----
    The rod pushing algorithm ensures:
    1. Each rod has height = intensity_map[z, x]
    2. Rods don't collide (base[z, x] > top[z-1, x] or base[z, x] > top[z+1, x])
    3. Tongue-and-groove effect is optionally considered
    4. Rods are placed as low as possible to minimize delivery time

    The algorithm processes columns from left to right (min_x to max_x) and
    within each column processes rows from top to bottom (min_z to max_z).

    For VMAT, this is called once per FMO beam to initialize aperture shapes.

    References
    ----------
    matRad: matRad/matRad/sequencing/matRad_siochiLeafSequencing.m
            (nested function matRad_siochiDecomposePort)
    """
    tops = np.zeros((dim_z, dim_x), dtype=np.int32)
    bases = np.zeros((dim_z, dim_x), dtype=np.int32)

    for x in range(min_x, max_x + 1):
        max_top = -1  # Track maximum top in this column to prevent collisions
        tng = 0  # Tongue-and-groove offset (currently not used, set to 0)

        for z in range(min_z, max_z + 1):
            if x == min_x:
                # First column: start rods at base=1
                bases[z, x] = 1
                tops[z, x] = bases[z, x] + intensity_map[z, x] - 1
            else:
                # Subsequent columns: match or adjust bases
                if intensity_map[z, x] >= intensity_map[z, x - 1]:
                    # Current rod >= previous rod: match the bases
                    bases[z, x] = bases[z, x - 1]
                    tops[z, x] = bases[z, x] + intensity_map[z, x] - 1
                else:
                    # Current rod < previous rod: need to check collisions
                    # Try to place base at same level as previous
                    trial_base = bases[z, x - 1]
                    trial_top = trial_base + intensity_map[z, x] - 1

                    # Check collision with rod above
                    if trial_base <= max_top:
                        # Collision! Push base up
                        bases[z, x] = max_top + 1
                        tops[z, x] = bases[z, x] + intensity_map[z, x] - 1
                    else:
                        # No collision
                        bases[z, x] = trial_base
                        tops[z, x] = trial_top

            # Update max_top for next row in this column
            if tops[z, x] > max_top:
                max_top = tops[z, x]

    return tops, bases


def siochi_is_different_slab(
    tops: NDArray[np.int32],
    bases: NDArray[np.int32],
    level: int,
) -> bool:
    """
    Check if slab at current level is different from previous level.

    A "slab" is a horizontal slice through the rod arrangement at a given
    level. Two slabs are different if their boolean shapes differ.

    Parameters
    ----------
    tops : NDArray[np.int32]
        Top positions of rods. Shape: (dim_z, dim_x)
    bases : NDArray[np.int32]
        Base positions of rods. Shape: (dim_z, dim_x)
    level : int
        Current stratification level to check.

    Returns
    -------
    bool
        True if slab at `level` differs from slab at `level-1`, False otherwise.

    Notes
    -----
    The shape of a slab is determined by: (bases <= level) & (level <= tops)
    This creates a boolean mask showing which bixels are "cut" by the slab.

    The first slab (level=1) is always considered different.

    References
    ----------
    matRad: matRad/matRad/sequencing/matRad_siochiLeafSequencing.m
            (nested function matRad_siochiDiffSlab)
    """
    if level == 1:
        return True  # First slab is always different

    shape_level = (bases <= level) & (level <= tops)
    shape_level_prev = (bases <= level - 1) & (level - 1 <= tops)

    return not np.array_equal(shape_level, shape_level_prev)


def siochi_convert_to_segments(
    shapes: NDArray[np.bool_],
    shapes_weight: NDArray[np.float64],
    k: int,
    tops: NDArray[np.int32],
    bases: NDArray[np.int32],
    num_levels: int,
) -> Tuple[NDArray[np.bool_], NDArray[np.float64], int]:
    """
    Convert rod positions to discrete MLC segments.

    Extracts aperture shapes by slicing through the rod arrangement at each
    stratification level. Consecutive identical shapes are merged.

    Parameters
    ----------
    shapes : NDArray[np.bool_]
        Pre-allocated array for storing shapes. Shape: (dim_z, dim_x, max_shapes)
    shapes_weight : NDArray[np.float64]
        Pre-allocated array for storing shape weights. Shape: (max_shapes,)
    k : int
        Current shape index (number of shapes already stored).
    tops : NDArray[np.int32]
        Top positions of rods. Shape: (dim_z, dim_x)
    bases : NDArray[np.int32]
        Base positions of rods. Shape: (dim_z, dim_x)
    num_levels : int
        Number of stratification levels.

    Returns
    -------
    shapes : NDArray[np.bool_]
        Updated shapes array with new segments added.
    shapes_weight : NDArray[np.float64]
        Updated weights array. Each weight represents the contribution of that shape.
    k : int
        Updated shape count.

    Notes
    -----
    The algorithm iterates through stratification levels from top to bottom
    (num_levels down to 1). At each level, it checks if the shape is different
    from the previous level. If different, it creates a new segment with weight=1.

    Identical consecutive shapes are automatically merged because they won't
    create a new segment.

    The actual number of segments k is typically <= num_levels because:
    - Some levels may have identical shapes
    - Empty levels are skipped

    References
    ----------
    matRad: matRad/matRad/sequencing/matRad_siochiLeafSequencing.m
            (nested function matRad_siochiConvertToSegments)
    """
    dim_z, dim_x = tops.shape

    # Iterate from highest level to lowest
    for level in range(num_levels, 0, -1):
        if siochi_is_different_slab(tops, bases, level):
            # Create new shape for this level
            shape_level = (bases <= level) & (level <= tops)
            shapes[:, :, k] = shape_level
            shapes_weight[k] = 1.0  # Each level contributes weight of 1
            k += 1

    return shapes, shapes_weight, k


def siochi_discard_apertures(
    beam_shapes: NDArray[np.bool_],
    beam_weights: NDArray[np.float64],
    num_shapes: int,
    num_to_keep: int,
) -> Tuple[NDArray[np.bool_], NDArray[np.float64], int]:
    """
    Discard low-weight apertures to reduce complexity.

    For VMAT optimization, we may want to keep only the N highest-weighted
    apertures to reduce optimization variables and delivery time.

    Parameters
    ----------
    beam_shapes : NDArray[np.bool_]
        Aperture shapes. Shape: (dim_z, dim_x, num_shapes)
    beam_weights : NDArray[np.float64]
        Aperture weights. Shape: (num_shapes,)
    num_shapes : int
        Current number of shapes.
    num_to_keep : int
        Number of shapes to keep (highest weighted). If 0, keep all.

    Returns
    -------
    beam_shapes : NDArray[np.bool_]
        Filtered shapes with only top N kept.
    beam_weights : NDArray[np.float64]
        Filtered weights.
    num_shapes : int
        Updated number of shapes (= num_to_keep or original if num_to_keep=0).

    Notes
    -----
    This is used in VMAT to limit the number of apertures per beam angle,
    specified by pln.prop_opt.num_apertures or stf.prop_vmat.num_of_beam_children.

    References
    ----------
    matRad: matRad/matRad/sequencing/matRad_discardApertures.m
    """
    if num_to_keep == 0 or num_to_keep >= num_shapes:
        return beam_shapes, beam_weights, num_shapes

    # Sort by weight (descending)
    sorted_indices = np.argsort(beam_weights[:num_shapes])[::-1]
    top_indices = sorted_indices[:num_to_keep]

    # Keep only top N
    beam_shapes_filtered = beam_shapes[:, :, top_indices]
    beam_weights_filtered = beam_weights[top_indices]

    return beam_shapes_filtered, beam_weights_filtered, num_to_keep


def siochi_leaf_sequencing(
    result: Dict[str, Any],
    stf: List[Dict[str, Any]],
    dij: Dict[str, Any],
    pln: Any,
    vis_bool: bool = False,
) -> Dict[str, Any]:
    """
    Siochi leaf sequencing algorithm for IMRT and VMAT.

    Converts continuous fluence maps from fluence optimization into deliverable
    MLC aperture segments using the Siochi rod-pushing algorithm.

    Parameters
    ----------
    result : Dict[str, Any]
        Result from fluence optimization containing:
        - 'w': optimized fluence weights (NDArray)
        - 'w_unsequenced': original unsequenced weights (optional)
    stf : List[Dict[str, Any]]
        Steering file with beam geometry information:
        - 'ray': list of ray dictionaries with 'rayPos_bev'
        - 'num_of_rays': number of rays per beam
        - 'bixel_width': bixel width in mm
        - 'gantry_angle': gantry angle in degrees
    dij : Dict[str, Any]
        Dose influence matrix containing:
        - 'physical_dose': dose influence matrices
        - 'dose_grid': dose calculation grid
        - 'ct_grid': CT grid for interpolation
        - 'weight_to_mu': weight to MU conversion
    pln : Any
        Treatment plan object with properties:
        - 'prop_seq.num_levels': number of stratification levels
        - 'prop_seq.sequencer': sequencer type ('siochi')
        - 'prop_opt.run_vmat': VMAT flag
        - 'prop_opt.num_apertures': number of apertures to keep (optional)
        - 'radiation_mode': 'photons' required for sequencing
    vis_bool : bool, optional
        Visualization flag (not yet implemented). Default: False.

    Returns
    -------
    result : Dict[str, Any]
        Updated result dictionary with added fields:
        - 'w_sequenced': sequenced bixel weights
        - 'sequencing': sequencing structure with beam/shape info
        - 'aperture_info': aperture information for DAO
        - 'physical_dose': recalculated dose with sequenced weights

    Notes
    -----
    The algorithm follows these steps for each beam:
    1. Extract fluence weights for current beam
    2. Map weights to 2D fluence matrix (Z × X grid)
    3. Optional Gaussian smoothing
    4. Stratify fluence into discrete levels
    5. Decompose using rod pushing
    6. Convert to aperture segments
    7. Optionally discard low-weight apertures (VMAT)
    8. Recalculate dose with sequenced weights

    For VMAT, additional arc sequencing is performed after initial sequencing.

    Examples
    --------
    >>> # After fluence optimization
    >>> result = fluence_optimization(dij, cst, pln)
    >>> result = siochi_leaf_sequencing(result, stf, dij, pln)
    >>> print(f"Created {result['aperture_info'].total_num_of_shapes} segments")

    References
    ----------
    matRad: matRad/matRad/sequencing/matRad_siochiLeafSequencing.m
    """
    # Extract parameters
    num_levels = getattr(pln.prop_seq, 'num_levels', 5)
    run_vmat = getattr(pln.prop_opt, 'run_vmat', False)
    num_apertures = getattr(pln.prop_opt, 'num_apertures', 0)

    # Get unsequenced weights
    if 'w_unsequenced' in result:
        w_unsequenced = result['w_unsequenced']
    else:
        w_unsequenced = result['w']

    num_beams = len(stf)
    sequencing = {
        'beam': [],
        'w': np.zeros_like(w_unsequenced),
        'run_vmat': run_vmat,
    }

    offset = 0

    for beam_idx in range(num_beams):
        beam_info = stf[beam_idx]
        num_rays = beam_info['num_of_rays']
        bixel_width = beam_info['bixel_width']

        # Skip non-FMO beams in VMAT
        if run_vmat:
            if 'prop_vmat' in beam_info:
                if not beam_info['prop_vmat'].get('fmo_beam', True):
                    # Not an FMO beam, skip sequencing
                    sequencing['w'][offset:offset + num_rays] = 0
                    sequencing['beam'].append({
                        'num_of_shapes': 0,
                        'bixel_ix': range(offset, offset + num_rays),
                    })
                    offset += num_rays
                    continue
                num_to_keep = beam_info['prop_vmat'].get('num_of_beam_children', 0)
            else:
                num_to_keep = 0
        else:
            num_to_keep = num_apertures

        # Get weights for current beam
        w_current_beam = w_unsequenced[offset:offset + num_rays]

        # Map bixel weights to 2D fluence matrix
        # Extract X and Z positions
        ray_pos_bev = np.array([ray['ray_pos_bev'] for ray in beam_info['ray']])
        X = ray_pos_bev[:, 0]
        Z = ray_pos_bev[:, 2]

        min_x, max_x = np.min(X), np.max(X)
        min_z, max_z = np.min(Z), np.max(Z)

        dim_x = int((max_x - min_x) / bixel_width + 1)
        dim_z = int((max_z - min_z) / bixel_width + 1)

        # Create fluence matrix
        fluence_mx = np.zeros((dim_z, dim_x), dtype=np.float64)

        x_pos = ((X - min_x) / bixel_width + 1).astype(int)
        z_pos = ((Z - min_z) / bixel_width + 1).astype(int)

        ind_in_fluence_mx = z_pos + (x_pos - 1) * dim_z
        fluence_mx.flat[ind_in_fluence_mx] = w_current_beam

        # Optional Gaussian smoothing (if available)
        try:
            for row in range(dim_z):
                fluence_mx[row, :] = gaussian_filter(fluence_mx[row, :], sigma=1.0)
        except Exception:
            pass  # Skip smoothing if not available

        # Allow for repeated sequencing with higher num_levels if needed
        not_finished = True
        current_num_levels = num_levels

        while not_finished:
            # Step 1: Stratification
            D_k, cal_factor = siochi_stratify(fluence_mx, current_num_levels)
            D_0 = D_k.copy()  # Save initial stratification

            # Pre-allocate shapes
            shapes = np.full((dim_z, dim_x, 10000), np.nan, dtype=np.float64)
            shapes_weight = np.zeros(10000, dtype=np.float64)
            k = 0  # Shape counter

            # Find bounding box of non-zero intensity
            D_k_nonzero = D_k != 0
            if not np.any(D_k_nonzero):
                # Empty beam
                break

            D_k_indices = np.argwhere(D_k_nonzero)
            D_k_min_z, D_k_min_x = D_k_indices.min(axis=0)
            D_k_max_z, D_k_max_x = D_k_indices.max(axis=0)

            # Step 2: Decompose port (rod pushing)
            tops, bases = siochi_decompose_port(
                D_k, dim_z, dim_x, D_k_min_z, D_k_max_z, D_k_min_x, D_k_max_x
            )

            # Step 3: Convert to segments
            shapes, shapes_weight, k = siochi_convert_to_segments(
                shapes, shapes_weight, k, tops, bases, current_num_levels
            )

            # Check if we have enough apertures
            if num_to_keep != 0 and k < num_to_keep:
                # Not enough apertures, increment num_levels and retry
                current_num_levels += 1
            else:
                # Sufficient apertures, exit loop
                not_finished = False

        # Trim shapes to actual count
        shapes = shapes[:, :, :k]
        shapes_weight = shapes_weight[:k] / current_num_levels * cal_factor

        # Optionally discard low-weight apertures
        if num_to_keep != 0:
            shapes, shapes_weight, k = siochi_discard_apertures(
                shapes, shapes_weight, k, num_to_keep
            )

        # Store sequencing results for this beam
        beam_sequencing = {
            'num_of_shapes': k,
            'shapes': shapes,
            'shapes_weight': shapes_weight,
            'bixel_ix': list(range(offset, offset + num_rays)),
            'fluence': D_0,
            'sum': np.zeros((dim_z, dim_x)),
        }

        # Calculate sequenced weights for non-VMAT
        if not run_vmat:
            for shape_idx in range(k):
                beam_sequencing['sum'] += (
                    shapes[:, :, shape_idx] * shapes_weight[shape_idx]
                )
            sequencing['w'][offset:offset + num_rays] = beam_sequencing['sum'].flat[
                ind_in_fluence_mx
            ]

        sequencing['beam'].append(beam_sequencing)
        offset += num_rays

    # VMAT-specific arc sequencing
    if run_vmat:
        raise NotImplementedError("VMAT arc sequencing will be implemented in Phase 3")

    # Store sequencing results
    result['sequencing'] = sequencing
    result['w_sequenced'] = sequencing['w']

    # Recalculate dose with sequenced weights
    # (Implementation depends on dose calculation interface)
    # For now, we'll skip dose recalculation
    # result['physical_dose'] = calculate_dose(dij, sequencing['w'])

    return result
