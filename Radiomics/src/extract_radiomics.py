#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd
import numpy as np
import SimpleITK as sitk
import pyradiomics
from pyradiomics import featureextractor
radiomics = pyradiomics
import logging
import yaml
import six
from datetime import datetime
import time
import tempfile
import shutil
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "7")

try:
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(8)
    print("Set SimpleITK to use 8 threads for parallel processing")
except:
    print("Could not set SimpleITK thread count")

def setup_logging(output_dir, level=logging.INFO):
    log_file = os.path.join(output_dir, 'pyradiomics_extraction.log')
    radiomics.setVerbosity(level)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('pyradiomics_extraction')

def create_pyradiomics_params(config_path):
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    if 'extraction' in config:
        extraction_params = config['extraction']
    else:
        extraction_params = config
    return extraction_params

def create_lung_mask_from_lobe_mask(lobe_mask_path, lung_type='left', output_dir=None):
    lobe_mask = sitk.ReadImage(lobe_mask_path)
    lobe_array = sitk.GetArrayFromImage(lobe_mask)
    if lung_type == 'left':
        lung_array = ((lobe_array == 1) | (lobe_array == 2)).astype(np.uint8)
    elif lung_type == 'right':
        lung_array = ((lobe_array == 3) | (lobe_array == 4) | (lobe_array == 5)).astype(np.uint8)
    else:
        raise ValueError(f"Invalid lung_type: {lung_type}. Must be 'left' or 'right'")
    lung_mask = sitk.GetImageFromArray(lung_array)
    lung_mask.CopyInformation(lobe_mask)
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(lobe_mask_path).replace('.nii.gz', '').replace('.nii', '')
    lung_mask_path = os.path.join(output_dir, f"{base_name}_{lung_type}_lung.nii.gz")
    sitk.WriteImage(lung_mask, lung_mask_path)
    return lung_mask_path

def create_images_csv_from_lobe_masks(lobe_masks_dir, original_images_dir, output_csv_path, temp_mask_dir):
    print(f"Creating images.csv from lobe masks directory: {lobe_masks_dir}")
    image_mask_pairs = []
    for patient_dir in sorted(os.listdir(original_images_dir)):
        patient_path = os.path.join(original_images_dir, patient_dir)
        lobe_mask_path = os.path.join(lobe_masks_dir, patient_dir)
        if not os.path.isdir(patient_path) or not os.path.exists(lobe_mask_path):
            continue
        print(f"Processing patient: {patient_dir}")
        for filename in sorted(os.listdir(patient_path)):
            if not filename.endswith('.nii.gz'):
                continue
            file_id = filename.replace('.nii.gz', '')
            image_path = os.path.join(patient_path, filename)
            lobe_mask_file = os.path.join(lobe_mask_path, filename)
            if not os.path.exists(lobe_mask_file):
                continue
            patient_temp_dir = os.path.join(temp_mask_dir, patient_dir)
            os.makedirs(patient_temp_dir, exist_ok=True)
            try:
                left_lung_mask_path = create_lung_mask_from_lobe_mask(lobe_mask_file, 'left', patient_temp_dir)
                right_lung_mask_path = create_lung_mask_from_lobe_mask(lobe_mask_file, 'right', patient_temp_dir)
                image_mask_pairs.append({
                    'Patient': patient_dir,
                    'Filename': file_id,
                    'Image': image_path,
                    'LeftLungMask': left_lung_mask_path,
                    'RightLungMask': right_lung_mask_path,
                    'LobeMask': lobe_mask_file
                })
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                continue
        if len(image_mask_pairs) % 10 == 0:
            print(f"  Processed {len(image_mask_pairs)} image-mask pairs...")
    if image_mask_pairs:
        df = pd.DataFrame(image_mask_pairs)
        df.to_csv(output_csv_path, index=False)
        print(f"Created images.csv with {len(df)} entries at {output_csv_path}")
        return df
    else:
        print("No image-mask pairs found!")
        return None

def extract_features(images_path, output_dir, params, batch_size=10, resume=True, label=None):
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    logger.info(f"Starting PyRadiomics extraction with parameters: {params}")
    images_df = pd.read_csv(images_path)
    logger.info(f"Loaded {len(images_df)} image-mask pairs from {images_path}")
    if label is not None:
        if 'setting' not in params:
            params['setting'] = {}
        params['setting']['label'] = label
        logger.info(f"Extracting features for label {label}")
    results_path = os.path.join(output_dir, 'radiomics_features.csv')
    if resume and os.path.exists(results_path):
        try:
            existing_features = pd.read_csv(results_path)
            logger.info(f"Resuming from existing results with {len(existing_features)} entries")
            processed_files = set(existing_features['Filename'].unique())
            images_df = images_df[~images_df['Filename'].isin(processed_files)]
            logger.info(f"Remaining {len(images_df)} files to process")
        except Exception as e:
            logger.error(f"Error loading existing results: {e}. Starting from beginning.")
            existing_features = None
    else:
        existing_features = None
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    if 'setting' in params:
        for setting_key, setting_value in params['setting'].items():
            extractor.settings[setting_key] = setting_value
    if 'imageType' in params:
        for image_type, image_settings in params['imageType'].items():
            extractor.enableImageTypeByName(image_type, **image_settings)
    if 'featureClass' in params:
        for feature_class, feature_list in params['featureClass'].items():
            if feature_list:
                extractor.enableFeatureClassByName(feature_class, feature_list)
            else:
                extractor.enableFeatureClassByName(feature_class)
    logger.info("Initialized PyRadiomics feature extractor")
    if len(images_df) == 0:
        logger.info("No images to process!")
        return
    start_time = time.time()
    all_results = []
    total_batches = (len(images_df) + batch_size - 1) // batch_size
    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, len(images_df))
        batch_df = images_df.iloc[batch_start:batch_end]
        logger.info(f"Processing batch {batch_num+1}/{total_batches} ({batch_end-batch_start} images)")
        batch_results = []
        for idx, row in batch_df.iterrows():
            image_path = row['Image']
            mask_path = row['Mask']
            patient_id = row['Patient']
            filename = row['Filename']
            try:
                logger.info(f"Processing {filename} (Patient: {patient_id})")
                result = extractor.execute(image_path, mask_path)
                feature_row = {
                    'Patient': patient_id,
                    'Filename': filename,
                    'Image': image_path,
                    'Mask': mask_path
                }
                for feature_name, feature_value in six.iteritems(result):
                    if feature_name.startswith('diagnostics_'):
                        continue
                    feature_row[feature_name] = feature_value
                batch_results.append(feature_row)
                feature_count = sum(1 for k in result.keys() if not k.startswith('diagnostics_'))
                logger.info(f"Successfully extracted {feature_count} features for {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                batch_results.append({
                    'Patient': patient_id,
                    'Filename': filename,
                    'Image': image_path,
                    'Mask': mask_path,
                    'error': str(e)
                })
        if batch_results:
            batch_df = pd.DataFrame(batch_results)
            all_results.append(batch_df)
            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                if existing_features is not None:
                    combined_results = pd.concat([existing_features, combined_results], ignore_index=True)
                combined_results.to_csv(results_path, index=False)
                logger.info(f"Saved intermediate results with {len(combined_results)} entries")
        elapsed_time = time.time() - start_time
        images_processed = batch_end
        images_per_second = images_processed / elapsed_time if elapsed_time > 0 else 0
        logger.info(f"Progress: {images_processed}/{len(images_df)} images processed")
        logger.info(f"Processing speed: {images_per_second:.2f} images/sec")
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        if existing_features is not None:
            final_results = pd.concat([existing_features, final_results], ignore_index=True)
        final_results.to_csv(results_path, index=False)
        logger.info(f"Saved final results with {len(final_results)} entries to {results_path}")
        return final_results
    else:
        logger.warning("No results generated!")
        return None

def extract_lobe_features(lobe_masks_dir, original_images_dir, output_base_dir, params, batch_size=10, resume=True, labels_path=None):
    os.makedirs(output_base_dir, exist_ok=True)
    images_csv_path = os.path.join(output_base_dir, 'images.csv')
    images_df = pd.DataFrame()
    for patient_dir in sorted(os.listdir(original_images_dir)):
        patient_path = os.path.join(original_images_dir, patient_dir)
        if os.path.isdir(patient_path):
            for filename in sorted(os.listdir(patient_path)):
                if filename.endswith('.nii.gz'):
                    file_id = filename.replace('.nii.gz', '')
                    image_path = os.path.join(patient_path, filename)
                    mask_path = os.path.join(lobe_masks_dir, patient_dir, filename)
                    if os.path.exists(mask_path):
                        images_df = pd.concat([images_df, pd.DataFrame([{
                            'Patient': patient_dir,
                            'Filename': file_id,
                            'Image': image_path,
                            'Mask': mask_path
                        }])], ignore_index=True)
    if len(images_df) == 0:
        print("No image-mask pairs found!")
        return None
    images_df.to_csv(images_csv_path, index=False)
    print(f"Created images.csv with {len(images_df)} entries")
    all_lobe_features = []
    for lobe_num in range(1, 6):
        print(f"\nProcessing Lobe {lobe_num}")
        lobe_params = params.copy()
        if 'setting' not in lobe_params:
            lobe_params['setting'] = {}
        else:
            lobe_params['setting'] = lobe_params['setting'].copy()
        lobe_params['setting']['label'] = lobe_num
        lobe_output_dir = os.path.join(output_base_dir, f"lobe_{lobe_num}")
        os.makedirs(lobe_output_dir, exist_ok=True)
        lobe_images_csv = os.path.join(lobe_output_dir, 'images.csv')
        images_df.to_csv(lobe_images_csv, index=False)
        lobe_features = extract_features(lobe_images_csv, lobe_output_dir, lobe_params, batch_size=batch_size, resume=resume, label=lobe_num)
        if lobe_features is not None:
            lobe_features['Region'] = f"Lobe {lobe_num}"
            lobe_csv_path = os.path.join(lobe_output_dir, f'lobe_{lobe_num}_features.csv')
            lobe_features.to_csv(lobe_csv_path, index=False)
            print(f"Saved features for Lobe {lobe_num} to {lobe_csv_path}")
            all_lobe_features.append(lobe_features)
    if all_lobe_features:
        combined_features = pd.concat(all_lobe_features, ignore_index=True)
        combined_dir = os.path.join(output_base_dir, 'combined_lobe_features')
        os.makedirs(combined_dir, exist_ok=True)
        combined_path = os.path.join(combined_dir, 'all_lobe_features.csv')
        combined_features.to_csv(combined_path, index=False)
        print(f"Saved combined features to {combined_path}")
        return combined_features
    return None

def extract_lung_features(lobe_masks_dir, original_images_dir, output_base_dir, params, batch_size=10, resume=True, labels_path=None):
    os.makedirs(output_base_dir, exist_ok=True)
    temp_mask_dir = os.path.join(output_base_dir, 'temp_lung_masks')
    os.makedirs(temp_mask_dir, exist_ok=True)
    images_csv_path = os.path.join(output_base_dir, 'images.csv')
    images_df = create_images_csv_from_lobe_masks(lobe_masks_dir, original_images_dir, images_csv_path, temp_mask_dir)
    if images_df is None:
        print("Error: Could not create images.csv file")
        return None
    lung_regions = {
        "left_lung": {
            "name": "Left Lung",
            "label": 1,
            "mask_column": "LeftLungMask",
            "output_dir": os.path.join(output_base_dir, "left_lung")
        },
        "right_lung": {
            "name": "Right Lung",
            "label": 1,
            "mask_column": "RightLungMask",
            "output_dir": os.path.join(output_base_dir, "right_lung")
        }
    }
    all_lung_features = []
    for region_key, region_info in lung_regions.items():
        print(f"\nProcessing {region_info['name']}")
        lung_params = params.copy()
        if 'setting' not in lung_params:
            lung_params['setting'] = {}
        else:
            lung_params['setting'] = lung_params['setting'].copy()
        lung_params['setting']['label'] = region_info['label']
        lung_output_dir = region_info['output_dir']
        os.makedirs(lung_output_dir, exist_ok=True)
        lung_images_csv = os.path.join(lung_output_dir, 'images.csv')
        lung_images_df = images_df[['Patient', 'Filename', 'Image', region_info['mask_column']]].copy()
        lung_images_df.rename(columns={region_info['mask_column']: 'Mask'}, inplace=True)
        lung_images_df.to_csv(lung_images_csv, index=False)
        lung_features = extract_features(lung_images_csv, lung_output_dir, lung_params, batch_size=batch_size, resume=resume, label=region_info['label'])
        if lung_features is not None:
            lung_features['Region'] = region_info['name']
            lung_csv_path = os.path.join(lung_output_dir, 'radiomics_features.csv')
            lung_features.to_csv(lung_csv_path, index=False)
            print(f"Saved features for {region_info['name']} to {lung_csv_path}")
            all_lung_features.append(lung_features)
    if all_lung_features:
        combined_features = pd.concat(all_lung_features, ignore_index=True)
        combined_dir = os.path.join(output_base_dir, 'combined_lung_features')
        os.makedirs(combined_dir, exist_ok=True)
        combined_path = os.path.join(combined_dir, 'left_right_lung_features.csv')
        combined_features.to_csv(combined_path, index=False)
        print(f"Saved combined features to {combined_path}")
        if labels_path and os.path.exists(labels_path):
            merge_features_with_labels(combined_path, labels_path, combined_dir)
        print(f"\nCleaning up temporary lung masks in {temp_mask_dir}")
        try:
            shutil.rmtree(temp_mask_dir)
        except Exception as e:
            print(f"Warning: Could not remove temporary directory: {e}")
        return combined_features
    return None

def merge_features_with_labels(features_path, labels_path, output_dir):
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    print(f"Loaded {len(features_df)} feature entries and {len(labels_df)} label entries")
    if 'Filename' not in features_df.columns or 'Filename' not in labels_df.columns:
        if 'ScanID' in labels_df.columns:
            labels_df['Filename'] = labels_df['ScanID']
    if 'Filename' in features_df.columns and 'Filename' in labels_df.columns:
        merged_df = pd.merge(features_df, labels_df, on='Filename', how='left', suffixes=('', '_labels'))
        print(f"Merged features and labels: {len(merged_df)} rows")
        for col in merged_df.columns:
            if col.endswith('_labels') and col.replace('_labels', '') in merged_df.columns:
                orig_col = col.replace('_labels', '')
                if merged_df[col].notna().sum() > merged_df[orig_col].notna().sum():
                    merged_df[orig_col] = merged_df[col]
                merged_df = merged_df.drop(columns=[col])
        merged_path = os.path.join(output_dir, 'radiomics_features_with_labels.csv')
        merged_df.to_csv(merged_path, index=False)
        print(f"Saved merged features and labels to {merged_path}")
        return merged_df
    else:
        print("Error: Could not merge features and labels - missing Filename column")
        return None

def check_features_exist(radiomics_features_dir, expert_config):
    missing_regions = []
    existing_regions = []
    for expert_name, expert_type in zip(expert_config['expert_names'], expert_config['expert_types']):
        if expert_type == 'lobe':
            lobe_num = int(expert_name.split('_')[1])
            feature_file = os.path.join(radiomics_features_dir, f"lobe_{lobe_num}", "radiomics_features.csv")
            if os.path.exists(feature_file):
                existing_regions.append(expert_name)
            else:
                missing_regions.append(expert_name)
        elif expert_type == 'lung':
            lung_name = expert_name.lower().replace('_', '_')
            feature_file = os.path.join(radiomics_features_dir, lung_name, "radiomics_features.csv")
            if os.path.exists(feature_file):
                existing_regions.append(expert_name)
            else:
                missing_regions.append(expert_name)
    return existing_regions, missing_regions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpu', type=str, default=None)
    args = parser.parse_args()
    
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    num_experts = config['num_experts']
    expert_key = f"{num_experts}_experts"
    expert_config = config['expert_config'][expert_key]
    expert_config['num_experts'] = num_experts
    
    data_paths = config['data_paths']
    output_paths = config['output_paths']
    extraction_config = config['extraction']
    
    radiomics_features_dir = output_paths['radiomics_features_dir']
    lobe_masks_dir = data_paths['lobe_masks_dir']
    nifti_dir = data_paths['nifti_dir']
    pyradiomics_config = data_paths['pyradiomics_config']
    labels_file = data_paths['labels_file']
    batch_size = extraction_config['batch_size']
    resume = extraction_config['resume']
    
    existing_regions, missing_regions = check_features_exist(radiomics_features_dir, expert_config)
    
    if existing_regions:
        print(f"✅ Found existing features for: {', '.join(existing_regions)}")
    
    if missing_regions:
        print(f"⚠️ Missing features for: {', '.join(missing_regions)}")
        print("Extracting missing features...")
        
        params = create_pyradiomics_params(pyradiomics_config)
        
        needs_lobes = any('lobe' in r.lower() for r in missing_regions)
        needs_lungs = any('lung' in r.lower() for r in missing_regions)
        
        if needs_lobes:
            print("\nExtracting lobe features...")
            extract_lobe_features(lobe_masks_dir, nifti_dir, radiomics_features_dir, params, batch_size, resume, labels_file)
        
        if needs_lungs:
            print("\nExtracting lung features...")
            extract_lung_features(lobe_masks_dir, nifti_dir, radiomics_features_dir, params, batch_size, resume, labels_file)
        
        print("\n✅ Feature extraction completed!")
    else:
        print("✅ All required features already exist!")

if __name__ == "__main__":
    main()

