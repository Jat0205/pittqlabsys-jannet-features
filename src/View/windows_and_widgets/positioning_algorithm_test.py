import numpy as np
from typing import Tuple, List




def from_four_corners_to_DMT_or_DMNT(
       corners_microdrive: List[np.ndarray],
       reference_order: Tuple[str] = ("top_left", "top_right", "bottom_right", "bottom_left")
) -> np.ndarray:
   """
   Compute transformation matrix from microdrive coordinates to diamond coordinates
   using four corner points.


   Parameters:
   -----------
   corners_microdrive : List[np.ndarray]
       List of 4 points in microdrive coordinates in order: [top_left, top_right, bottom_right, bottom_left]
       Each point should be a numpy array of shape (2,) for (x, y)


   reference_order : Tuple[str], optional
       Order of the corners as they appear in the corners_microdrive list


   Returns:
   --------
   T : np.ndarray
       3x3 transformation matrix such that:
       [x_diamond, y_diamond, 1]^T = T @ [x_microdrive, y_microdrive, 1]^T
       from microdrive coordinate system to diamond coordinate system
       DMT is diamond microdrive transformation matrix
       DMNT is diamond microdrive new transformation matrix


   Notes:
   ------
   Diamond coordinate system definition:
   - Origin: at bottom_left corner
   - x-axis: from bottom_left to bottom_right
   - y-axis: from bottom_left to top_left
   This assumes the corners form a (possibly rotated) rectangle in microdrive coords.
   """


   if len(corners_microdrive) != 4:
       raise ValueError(f"Expected 4 corners, got {len(corners_microdrive)}")


   # Extract points in the specified order
   # Assuming input order matches reference_order
   bl_idx = reference_order.index("bottom_left")
   br_idx = reference_order.index("bottom_right")
   tl_idx = reference_order.index("top_left")


   bottom_left = np.array(corners_microdrive[bl_idx], dtype=float)
   bottom_right = np.array(corners_microdrive[br_idx], dtype=float)
   top_left = np.array(corners_microdrive[tl_idx], dtype=float)


   # Diamond coordinate system definition:
   # In diamond coords:
   # bottom_left = (0, 0)
   # bottom_right = (1, 0)
   # top_left = (0, 1)
   # top_right = (1, 1)


   # Source points in microdrive coordinates (homogeneous)
   src_points = np.array([
       [bottom_left[0], bottom_left[1], 1],
       [bottom_right[0], bottom_right[1], 1],
       [top_left[0], top_left[1], 1]
   ]).T  # Shape: (3, 3)


   # Destination points in diamond coordinates (homogeneous)
   dst_points = np.array([
       [0, 0, 1],
       [1, 0, 1],
       [0, 1, 1]
   ]).T  # Shape: (3, 3)


   # Solve for transformation matrix T such that: dst = T @ src
   # T is 3x3, we need T @ src = dst
   # T = dst @ inv(src)


   try:
       T = dst_points @ np.linalg.inv(src_points)
   except np.linalg.LinAlgError:
       raise ValueError("Corners are colinear or form a degenerate shape")


   # Verify with the fourth point
   top_right = np.array(corners_microdrive[reference_order.index("top_right")], dtype=float)
   tr_hom = np.array([top_right[0], top_right[1], 1])
   tr_diamond_pred = T @ tr_hom
   # Should be close to (1, 1, 1) after normalization
   tr_diamond_pred = tr_diamond_pred / tr_diamond_pred[2]


   # Check if the shape is reasonably rectangular
   expected = np.array([1, 1, 1])
   if np.linalg.norm(tr_diamond_pred - expected) > 1e-3:
       print(f"Warning: Fourth corner verification failed. "
             f"Expected (1, 1, 1), got ({tr_diamond_pred[0]:.3f}, {tr_diamond_pred[1]:.3f}, {tr_diamond_pred[2]:.3f})")
       print("This suggests corners don't form a proper rectangle or order is wrong.")


   return T




def from_DMT_and_MNV_old_get_DNV_old(M_point: np.ndarray, T_matrix: np.ndarray) -> np.ndarray:
   """
   Transform a point from microdrive to diamond coordinates using transformation matrix.
   DNV_old = DMT * MNV_old


   Parameters:
   -----------
   M_point : np.ndarray
       Point in microdrive coordinates, shape (2,) or (3,) (if homogeneous)
   T_matrix : np.ndarray
       3x3 transformation matrix


   Returns:
   --------
   D_point : np.ndarray
       Point in diamond coordinates, shape (2,)
   """
   if len(M_point) == 2:
       M_hom = np.array([M_point[0], M_point[1], 1.0])
   else:
       M_hom = np.array(M_point)


   D_hom = T_matrix @ M_hom
   D_hom = D_hom / D_hom[2]  # Normalize


   return D_hom[:2]

import numpy as np

# Test data
old_corners = [
    np.array([100, 200]),  # top_left
    np.array([300, 200]),  # top_right
    np.array([100, 400]),  # bottom_left
    np.array([300, 400])  # bottom_right
]

new_corners = [
    np.array([50, 100]),  # top_left
    np.array([250, 100]),  # top_right
    np.array([50, 300]),  # bottom_left
    np.array([250, 300])  # bottom_right
]

MNV_old = np.array([150, 300])
MNV_new_true = np.array([100, 200])  # The "answer" we're trying to compute
# Compute transformations
DMT = from_four_corners_to_DMT_or_DMNT(old_corners)
DMNT = from_four_corners_to_DMT_or_DMNT(new_corners)

# Our computation path:
# 1. DNV_old = DMT * MNV_old
DNV_old_hom = DMT @ np.array([MNV_old[0], MNV_old[1], 1])
DNV_old = DNV_old_hom[:2] / DNV_old_hom[2]
print(f"DNV_old: {DNV_old}")  # Should be (0.25, 0.5) in diamond coords

# 2. MNV_new = DMNT^{-1} * DNV_old
DMNT_inv = np.linalg.inv(DMNT)
MNV_new_pred_hom = DMNT_inv @ np.array([DNV_old[0], DNV_old[1], 1])
MNV_new_pred = MNV_new_pred_hom[:2] / MNV_new_pred_hom[2]
print(f"Predicted MNV_new: {MNV_new_pred}")
print(f"True MNV_new: {MNV_new_true}")
print(f"Error: {np.linalg.norm(MNV_new_pred - MNV_new_true)}")

# Direct method (one line):
MNV_new_direct = from_DMT_and_MNV_old_get_DNV_old(MNV_old, np.linalg.inv(DMNT) @ DMT)
print(f"Direct computation: {MNV_new_direct}")