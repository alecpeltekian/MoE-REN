import os
import nibabel as nib
import numpy as np
from tqdm import tqdm  # For progress bars

def process_all_ct_images(ct_folder, mask_folder, output_folder):
    """
    Process all CT images ensuring exact zero values in background.
    
    Args:
        ct_folder: Path to CT images
        mask_folder: Path to mask images
        output_folder: Path to save output
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Build list of all files to process
    files_to_process = []
    for patient_dir in os.listdir(ct_folder):
        patient_ct_dir = os.path.join(ct_folder, patient_dir)
        if not os.path.isdir(patient_ct_dir):
            continue
            
        for filename in os.listdir(patient_ct_dir):
            if filename.endswith('.nii.gz'):
                files_to_process.append(f"{patient_dir}/{filename}")
    
    print(f"Found {len(files_to_process)} files to process")
    
    # Stats tracking
    successful = 0
    failed = 0
    skipped = 0
    
    # Process all files with progress bar
    for file_path in tqdm(files_to_process, desc="Processing CT images"):
        patient_dir, filename = file_path.split('/', 1)
        
        # Setup paths
        patient_output_dir = os.path.join(output_folder, patient_dir)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        ct_path = os.path.join(ct_folder, patient_dir, filename)
        mask_path = os.path.join(mask_folder, patient_dir, filename)
        
        if not os.path.exists(ct_path) or not os.path.exists(mask_path):
            skipped += 1
            continue
            
        try:
            # Load original CT image
            ct_img = nib.load(ct_path)
            header = ct_img.header.copy()
            ct_data = ct_img.get_fdata()
            
            # Load mask
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata()
            
            # Create binary mask
            binary_mask = np.where(mask_data > 0, 1, 0)
            
            # Apply mask - preserve all original values inside lung
            segmented_data = ct_data.copy()
            segmented_data[binary_mask == 0] = 0  # Set outside to zero
            
            # Explicitly set data type to match original
            segmented_data = segmented_data.astype(np.float32)
            
            # Create new image with unscaled data
            output_img = nib.Nifti1Image(segmented_data, ct_img.affine, header)
            
            # Explicitly set header scaling to prevent altered zeros
            output_img.header['scl_slope'] = 1.0
            output_img.header['scl_inter'] = 0.0
            
            # Save with no scaling
            output_path = os.path.join(patient_output_dir, filename)
            nib.save(output_img, output_path)
            
            # Verify the saved file
            verify_img = nib.load(output_path)
            verify_data = verify_img.get_fdata()
            verify_outside = verify_data[binary_mask == 0]
            
            # Check for exact zeros (allow tiny float error)
            if np.all(np.abs(verify_outside) < 1e-10):
                successful += 1
            else:
                # Still count as processed but log the issue
                print(f"\nWarning: {file_path} - Background values not exactly zero after saving")
                print(f"  Min: {np.min(verify_outside)}, Max: {np.max(verify_outside)}")
                failed += 1
            
        except Exception as e:
            print(f"\nError processing {file_path}: {str(e)}")
            failed += 1
    
    # Print final summary
    print("\nProcessing Summary")
    print(f"Total files found: {len(files_to_process)}")
    print(f"Successfully processed: {successful}")
    print(f"Processed with issues: {failed}")
    print(f"Skipped (missing files): {skipped}")
    print("\nProcessing complete!")

# Run the processor
if __name__ == "__main__":
    ct_folder = '/data2/akp4895/1mm_OrientedImages'
    mask_folder = '/data2/akp4895/1mm_OrientedImagesSegmented'
    output_folder = '/data2/akp4895/MultipliedImagesClean'
    
    process_all_ct_images(ct_folder, mask_folder, output_folder)