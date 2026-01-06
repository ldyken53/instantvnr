#!/usr/bin/env python3
"""
Convert a 2048×2048×1920 uint8 raw volume (NRRD-style) into a
float32, normalized [0,1] voxel dataset on the unit cube, and save
as a structured VTK file using PyVista (ImageData).
"""

import numpy as np
import pyvista as pv
import os

def read_raw_volume(filename, shape, dtype=np.uint8, order='C'):
    """
    Reads a raw binary file into a NumPy array.
    """
    count = np.prod(shape)
    data = np.fromfile(filename, dtype=dtype, count=count)
    if data.size != count:
        raise IOError(f"Expected {count} elements, got {data.size}")
    return data.reshape(shape, order=order)

def normalize_volume(vol):
    """
    Linearly scales vol so its values lie in [0,1].
    """
    vmin, vmax = vol.min(), vol.max()
    if vmax == vmin:
        return np.zeros_like(vol, dtype=np.float32)
    return (vol - vmin) / (vmax - vmin)

def build_image_data(volume):
    """
    Builds a PyVista ImageData whose physical extents
    fit inside [0,1]^3 *while preserving the original shape*.
    The longest axis will span exactly 1.0; the others
    will be scaled proportionally.
    """
    nx, ny, nz = volume.shape

    # Find the largest dimension
    max_dim = max(nx - 1, ny - 1, nz - 1)
    # Use a *uniform* spacing so the longest axis = 1.0
    common_spacing = 1.0 / max_dim
    spacing = (common_spacing, common_spacing, common_spacing)

    origin = (0.0, 0.0, 0.0)
    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=spacing,
        origin=origin
    )
    grid.point_data["value"] = volume.ravel(order='C')
    return grid


def main():
    # --- Update these to match your header ---
    raw_file = 'testrm.raw'
    vtk_file = 'chame.vtk'
    shape    = (1024, 1024, 1080)  # as per sizes: X, Y, Z
    dtype    = np.float32           # input type is uint8
    order    = 'C'                # use 'F' if needed for Fortran-order
    
    if not os.path.isfile(raw_file):
        raise FileNotFoundError(f"Could not locate raw file: {raw_file}")
    
    vol_u8 = read_raw_volume(raw_file, shape, dtype=dtype, order=order)
    
    vol_f32 = vol_u8.astype(np.float32)
    
    print("3) Normalizing to [0,1]…")
    # vol_norm = normalize_volume(vol_f32)
    
    print("4) Building ImageData in [0,1]^3…")
    grid = build_image_data(vol_f32)
    
    print(f"5) Saving to VTK: {vtk_file} …")
    grid.save(vtk_file)
    
    print("Done!")

if __name__ == "__main__":
    main()
