#!/usr/bin/env python3
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys

def visualize_vertical_flip(nifti_path, output_dir="/data2/akp4895/RawImages"):
    if not os.path.exists(nifti_path):
        print(f"Error: File {nifti_path} does not exist.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        flipped_data = np.flip(data, axis=1)
        
        orientation = nib.orientations.aff2axcodes(img.affine)
        qform_code = img.header['qform_code']
        sform_code = img.header['sform_code']
        
        print(f"File: {nifti_path}")
        print(f"Dimensions: {data.shape}")
        print(f"Original Header Orientation: {orientation}")
        print(f"qform_code: {qform_code}, sform_code: {sform_code}")
        print(f"Data type: {data.dtype}")
        print(f"Value range: {np.min(data)} to {np.max(data)}")
        
        mid_x = data.shape[0] // 2
        mid_y = data.shape[1] // 2
        mid_z = data.shape[2] // 2
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axial_slice = data[:, :, mid_z]
        axes[0, 0].imshow(np.rot90(axial_slice), cmap='gray')
        axes[0, 0].set_title(f'Original Axial (Z={mid_z})')
        
        coronal_slice = data[:, mid_y, :]
        axes[0, 1].imshow(np.rot90(coronal_slice), cmap='gray')
        axes[0, 1].set_title(f'Original Coronal (Y={mid_y})')
        
        sagittal_slice = data[mid_x, :, :]
        axes[0, 2].imshow(np.rot90(sagittal_slice), cmap='gray')
        axes[0, 2].set_title(f'Original Sagittal (X={mid_x})')
        
        flipped_axial = flipped_data[:, :, mid_z]
        axes[1, 0].imshow(np.rot90(flipped_axial), cmap='gray')
        axes[1, 0].set_title(f'Y-Flipped Axial (Z={mid_z})')
        
        flipped_coronal = flipped_data[:, mid_y, :]
        axes[1, 1].imshow(np.rot90(flipped_coronal), cmap='gray')
        axes[1, 1].set_title(f'Y-Flipped Coronal (Y={mid_y})')
        
        flipped_sagittal = flipped_data[mid_x, :, :]
        axes[1, 2].imshow(np.rot90(flipped_sagittal), cmap='gray')
        axes[1, 2].set_title(f'Y-Flipped Sagittal (X={mid_x})')
        
        fig.suptitle(f'NIfTI Image: Original vs Vertically Flipped', fontsize=16)
        
        plt.tight_layout()
        
        base_filename = os.path.basename(nifti_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{base_filename}_vertical_flip.png")
        
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")
        plt.close()

    
        flip_names = ['X-Flipped', 'Y-Flipped', 'Z-Flipped', 'XY-Flipped', 'XZ-Flipped', 'YZ-Flipped', 'XYZ-Flipped']
        flip_data = [
            np.flip(data, axis=0),                             # X-Flipped
            np.flip(data, axis=1),                             # Y-Flipped
            np.flip(data, axis=2),                             # Z-Flipped
            np.flip(np.flip(data, axis=0), axis=1),            # XY-Flipped
            np.flip(np.flip(data, axis=0), axis=2),            # XZ-Flipped
            np.flip(np.flip(data, axis=1), axis=2),            # YZ-Flipped
            np.flip(np.flip(np.flip(data, axis=0), axis=1), axis=2)  # XYZ-Flipped
        ]
        
        fig, axes = plt.subplots(8, 3, figsize=(15, 24))
        
        axial_slice = data[:, :, mid_z]
        axes[0, 0].imshow(np.rot90(axial_slice), cmap='gray')
        axes[0, 0].set_title(f'Original Axial')
        
        coronal_slice = data[:, mid_y, :]
        axes[0, 1].imshow(np.rot90(coronal_slice), cmap='gray')
        axes[0, 1].set_title(f'Original Coronal')
        
        sagittal_slice = data[mid_x, :, :]
        axes[0, 2].imshow(np.rot90(sagittal_slice), cmap='gray')
        axes[0, 2].set_title(f'Original Sagittal')
        
        axes[0, 0].set_ylabel('Original', fontsize=12)
        
        for i, (name, vol) in enumerate(zip(flip_names, flip_data)):
            row = i + 1
            
            axes[row, 0].imshow(np.rot90(vol[:, :, mid_z]), cmap='gray')
            
            axes[row, 1].imshow(np.rot90(vol[:, mid_y, :]), cmap='gray')
            
            axes[row, 2].imshow(np.rot90(vol[mid_x, :, :]), cmap='gray')
            
            axes[row, 0].set_ylabel(name, fontsize=12)
        
        fig.suptitle(f'NIfTI Image: All Flip Combinations', fontsize=16)
        
        plt.tight_layout()
        
        all_flips_path = os.path.join(output_dir, f"{base_filename}_all_flips.png")
        
        plt.savefig(all_flips_path)
        print(f"All flips visualization saved to: {all_flips_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing {nifti_path}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_flipped.py /path/to/image.nii.gz")
        example_path = '/data2/akp4895/1mm_Images/1618/1618_2019_1.nii.gz'
        print(f"Example: python visualize_flipped.py {example_path}")
    else:
        nifti_path = sys.argv[1]
        visualize_vertical_flip(nifti_path)