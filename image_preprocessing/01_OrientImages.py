import os
import glob
import nibabel as nib
import numpy as np
from pathlib import Path
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation, io_orientation

def reorient_nifti_files(input_files, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for i, nifti_path in enumerate(input_files):
        try:
            rel_path = os.path.relpath(nifti_path, os.path.dirname(os.path.commonprefix(input_files)))
            output_subdir = os.path.dirname(os.path.join(output_dir, rel_path))
            
            Path(output_subdir).mkdir(parents=True, exist_ok=True)
            
            img = nib.load(nifti_path)
            
            current_orientation = nib.aff2axcodes(img.affine)
            current_orientation_str = ''.join(current_orientation)
            
            if current_orientation_str == 'RAS':
                output_path = os.path.join(output_dir, rel_path)
                nib.save(img, output_path)
                print(f"[{i+1}/{len(input_files)}] {nifti_path} already in RAS orientation, copied to output")
                continue
            
            orig_header = img.header.copy()
            data = img.get_fdata()
            
            ornt_src = axcodes2ornt(current_orientation)
            ornt_tgt = axcodes2ornt(("R", "A", "S"))
            transform = ornt_transform(ornt_src, ornt_tgt)
            
            reoriented_data = apply_orientation(data, transform)
            
            new_affine = nib.orientations.inv_ornt_aff(transform, data.shape)
            reoriented_affine = np.dot(img.affine, new_affine)
            
            reoriented_img = nib.Nifti1Image(reoriented_data, reoriented_affine, header=orig_header)
            
            for field in ['scl_slope', 'scl_inter', 'cal_max', 'cal_min', 
                          'glmax', 'glmin', 'pixdim']:
                if field in orig_header:
                    try:
                        reoriented_img.header[field] = orig_header[field]
                    except:
                        pass
            
            output_path = os.path.join(output_dir, rel_path)
            
            nib.save(reoriented_img, output_path)
            
            print(f"[{i+1}/{len(input_files)}] Reoriented {nifti_path} from {current_orientation_str} to RAS")
            
        except Exception as e:
            print(f"Error processing {nifti_path}: {str(e)}")

if __name__ == "__main__":
    input_directory = "/data2/akp4895/RawImages"
    output_directory = "/data2/akp4895/OrientedImages"
    
    all_nifti_files = glob.glob(os.path.join(input_directory, "**", "*.nii.gz"), recursive=True)
    
    reorient_nifti_files(all_nifti_files, output_directory)
    
    print("Reorientation of example files complete.")