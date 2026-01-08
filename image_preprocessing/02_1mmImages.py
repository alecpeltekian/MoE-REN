#!/usr/bin/env python3

import os
import glob
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom

def resample_nifti(nifti_file, output_file, new_slice_thickness=1.0):
    try:
        img = nib.load(nifti_file)
        data = img.get_fdata()
        header = img.header
        
        current_voxel_dims = header.get_zooms()
        
        zoom_factors = (
            current_voxel_dims[0] / new_slice_thickness,
            current_voxel_dims[1] / new_slice_thickness,
            current_voxel_dims[2] / new_slice_thickness
        )
        
        resampled_data = zoom(data, zoom_factors, order=3)
        
        new_affine = np.copy(img.affine)
        scaling = np.eye(4)
        scaling[0, 0] = new_slice_thickness / current_voxel_dims[0]
        scaling[1, 1] = new_slice_thickness / current_voxel_dims[1]
        scaling[2, 2] = new_slice_thickness / current_voxel_dims[2]
        new_affine = new_affine @ scaling
        
        new_header = header.copy()
        new_header.set_zooms((new_slice_thickness, new_slice_thickness, new_slice_thickness))
        
        resampled_img = nib.Nifti1Image(resampled_data, affine=new_affine, header=new_header)
        
        nib.save(resampled_img, output_file)
        return True
    except Exception as e:
        print(f"Error resampling {nifti_file}: {str(e)}")
        return False

def resample_all_nifti_files(input_dir, output_dir, new_slice_thickness=1.0):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    nifti_pattern = os.path.join(input_dir, "**", "*.nii*")
    all_nifti_files = glob.glob(nifti_pattern, recursive=True)
    
    success_count = 0
    error_count = 0
    
    for i, nifti_path in enumerate(all_nifti_files):
        rel_path = os.path.relpath(nifti_path, input_dir)
        output_subdir = os.path.dirname(os.path.join(output_dir, rel_path))
        
        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        
        output_path = os.path.join(output_dir, rel_path)
        
        print(f"[{i+1}/{len(all_nifti_files)}] Resampling {nifti_path} to {new_slice_thickness}mm")
        
        if resample_nifti(nifti_path, output_path, new_slice_thickness):
            success_count += 1
        else:
            error_count += 1
    
    print(f"Resampling complete: {success_count} files processed successfully, {error_count} errors")

if __name__ == "__main__":
    input_directory = "/data2/akp4895/OrientedImages"
    output_directory = "/data2/akp4895/1mm_OrientedImages"
    
    resample_all_nifti_files(input_directory, output_directory, new_slice_thickness=1.0)