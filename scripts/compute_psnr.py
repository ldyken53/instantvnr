#!/usr/bin/env python3
"""
Compute PSNR of an error residual (against all zeroes) from NRRD format files.
"""

import numpy as np
import argparse
import os
import sys
import torch


def parse_nhdr(nhdr_path):
    """Parse NRRD header file and extract metadata."""
    metadata = {}
    
    with open(nhdr_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip().lower()] = value.strip()
    
    return metadata


def get_numpy_dtype(nrrd_type, endian):
    """Convert NRRD type to numpy dtype."""
    type_map = {
        'float': 'f4',
        'double': 'f8',
        'int': 'i4',
        'uint': 'u4',
        'short': 'i2',
        'ushort': 'u2',
        'char': 'i1',
        'uchar': 'u1',
    }
    
    if nrrd_type not in type_map:
        raise ValueError(f"Unsupported NRRD type: {nrrd_type}")
    
    dtype = type_map[nrrd_type]
    
    # Add endianness
    if endian == 'little':
        dtype = '<' + dtype
    elif endian == 'big':
        dtype = '>' + dtype
    
    return np.dtype(dtype)


def compute_psnr(residual):
    """
    Compute PSNR of residual against all zeroes.
    
    PSNR = 10 * log10(MAX^2 / MSE)
    where MSE = mean(residual^2) since we compare against zeros
    """
    # Compute MSE (mean squared error against zeros)
    mse = np.mean(residual ** 2)
    
    if mse == 0:
        return float('inf')  # Perfect match
    
    # Use the maximum absolute value in the residual as MAX
    # max_val = np.max(np.abs(residual))
    max_val = 1.0

    if max_val == 0:
        return float('inf')  # All zeros
    
    # Compute PSNR
    psnr = 10 * np.log10(max_val ** 2 / mse)
    
    return psnr


def main():
    parser = argparse.ArgumentParser(
        description='Compute PSNR of error residual against all zeroes from NRRD files'
    )
    parser.add_argument('nhdr_file', help='Path to .nhdr header file')
    parser.add_argument('raw_file', help='Path to .raw data file')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.nhdr_file):
        print(f"Error: NHDR file not found: {args.nhdr_file}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.raw_file):
        print(f"Error: RAW file not found: {args.raw_file}", file=sys.stderr)
        sys.exit(1)
    
    # Parse header
    print(f"Reading header: {args.nhdr_file}")
    metadata = parse_nhdr(args.nhdr_file)
    
    # Extract necessary information
    try:
        data_type = metadata['type']
        dimension = int(metadata['dimension'])
        sizes = [int(s) for s in metadata['sizes'].split()]
        encoding = metadata['encoding']
        endian = metadata['endian']
    except KeyError as e:
        print(f"Error: Missing required field in header: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Verify encoding
    if encoding != 'raw':
        print(f"Error: Only 'raw' encoding is supported, got: {encoding}", file=sys.stderr)
        sys.exit(1)
    
    # Get numpy dtype
    dtype = get_numpy_dtype(data_type, endian)
    
    print(f"Data type: {data_type} ({dtype})")
    print(f"Dimensions: {dimension}D")
    print(f"Sizes: {' x '.join(map(str, sizes))}")
    print(f"Total voxels: {np.prod(sizes)}")
    
    # Read raw data
    print(f"\nReading raw data: {args.raw_file}")
    residual = np.fromfile(args.raw_file, dtype=dtype)
    
    # Verify size
    expected_size = np.prod(sizes)
    if len(residual) != expected_size:
        print(f"Warning: Data size mismatch. Expected {expected_size}, got {len(residual)}", 
              file=sys.stderr)
    
    # Reshape to volume
    residual = residual.reshape(sizes)
    
    # Compute statistics
    print(f"\nResidual statistics:")
    print(f"  Min: {np.min(residual):.6e}")
    print(f"  Max: {np.max(residual):.6e}")
    print(f"  Mean: {np.mean(residual):.6e}")
    print(f"  Std: {np.std(residual):.6e}")
    print(f"  RMS: {np.sqrt(np.mean(residual**2)):.6e}")
    
    # Compute PSNR
    psnr = compute_psnr(residual)
    
    print(f"\nPSNR (against all zeroes): {psnr:.2f} dB")

    cells = torch.from_numpy(residual)
    mse = torch.mean((cells) ** 2)
    psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
    print(f"\nPSNR2 (against all zeroes): {psnr:.2f} dB")


if __name__ == '__main__':
    main()