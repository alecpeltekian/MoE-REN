
import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import logging
import yaml
import six
from datetime import datetime
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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

def extract_features(images_path, output_dir, params, batch_size=10, resume=True, label=None):
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Starting PyRadiomics extraction with parameters: {params}")
    logger.info(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

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
            logger.info(f"Found {len(processed_files)} already processed files")

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
            logger.info(f"Applied setting: {setting_key} = {setting_value}")

    if 'imageType' in params:
        for image_type, image_settings in params['imageType'].items():
            extractor.enableImageTypeByName(image_type, **image_settings)
            logger.info(f"Enabled image type: {image_type} with settings {image_settings}")

    if 'featureClass' in params:
        for feature_class, feature_list in params['featureClass'].items():
            if feature_list:
                extractor.enableFeatureClassByName(feature_class, feature_list)
            else:
                extractor.enableFeatureClassByName(feature_class)
            logger.info(f"Enabled feature class: {feature_class}")

    logger.info("Initialized PyRadiomics feature extractor")
    logger.info(f"Final settings: {extractor.settings}")
    logger.info(f"Enabled image types: {extractor.enabledImagetypes}")
    logger.info(f"Enabled features: {extractor.enabledFeatures}")

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
        estimated_remaining = (len(images_df) - images_processed) / images_per_second if images_per_second > 0 else 0

        logger.info(f"Progress: {images_processed}/{len(images_df)} images processed")
        logger.info(f"Processing speed: {images_per_second:.2f} images/sec")
        logger.info(f"Estimated time remaining: {estimated_remaining/60:.1f} minutes")

    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)

        if existing_features is not None:
            final_results = pd.concat([existing_features, final_results], ignore_index=True)

        final_results.to_csv(results_path, index=False)
        logger.info(f"Saved final results with {len(final_results)} entries to {results_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_path = os.path.join(output_dir, f'radiomics_features_{timestamp}.csv')
        final_results.to_csv(timestamped_path, index=False)
        logger.info(f"Saved timestamped copy to {timestamped_path}")

        return final_results
    else:
        logger.warning("No results generated!")
        return None

def create_images_csv_from_lung_segmentation(segmented_dir, original_images_dir, output_csv_path):
    """
    Create an images.csv file from the new lung segmentation directory structure.
    Assumes segmented_dir and original_images_dir contain patient ID subdirectories.
    """
    print(f"Creating images.csv from segmented directory: {segmented_dir}")
    print(f"Original images directory: {original_images_dir}")

    image_mask_pairs = []


    if os.path.exists(segmented_dir):
        for patient_id in os.listdir(segmented_dir):
            patient_seg_dir = os.path.join(segmented_dir, patient_id)
            patient_orig_dir = os.path.join(original_images_dir, patient_id)


            if not os.path.isdir(patient_seg_dir):
                continue

            print(f"Processing patient: {patient_id}")


            if not os.path.exists(patient_orig_dir):
                print(f"Warning: No original directory found for patient {patient_id}")
                continue


            seg_files = []
            if os.path.exists(patient_seg_dir):
                for filename in os.listdir(patient_seg_dir):
                    if filename.endswith('.nii.gz') or filename.endswith('.nii'):
                        seg_files.append(filename)


            orig_files = []
            if os.path.exists(patient_orig_dir):
                for filename in os.listdir(patient_orig_dir):
                    if filename.endswith('.nii.gz') or filename.endswith('.nii'):
                        orig_files.append(filename)

            print(f"  Found {len(seg_files)} segmentation files and {len(orig_files)} original files")


            if seg_files and orig_files:



                for seg_file in seg_files:
                    seg_path = os.path.join(patient_seg_dir, seg_file)


                    matched_orig = None


                    if seg_file in orig_files:
                        matched_orig = os.path.join(patient_orig_dir, seg_file)


                    if not matched_orig:
                        seg_basename = seg_file.replace('.nii.gz', '').replace('.nii', '')
                        for orig_file in orig_files:
                            orig_basename = orig_file.replace('.nii.gz', '').replace('.nii', '')
                            if seg_basename == orig_basename:
                                matched_orig = os.path.join(patient_orig_dir, orig_file)
                                break


                    if not matched_orig and len(orig_files) == 1:
                        matched_orig = os.path.join(patient_orig_dir, orig_files[0])

                    if matched_orig:
                        image_mask_pairs.append({
                            'Patient': patient_id,
                            'Filename': f"{patient_id}_{seg_file}",
                            'Image': matched_orig,
                            'Mask': seg_path
                        })
                        print(f"  Matched: {seg_file} -> {os.path.basename(matched_orig)}")
                    else:
                        print(f"  Warning: No matching original file found for {seg_file}")
            else:
                print(f"  Warning: Missing files for patient {patient_id}")


    if image_mask_pairs:
        df = pd.DataFrame(image_mask_pairs)
        df.to_csv(output_csv_path, index=False)
        print(f"Created images.csv with {len(df)} entries at {output_csv_path}")
        return df
    else:
        print("No image-mask pairs found!")
        return None

def extract_features_left_right_lung(segmented_dir, original_images_dir, output_base_dir, params, batch_size=10, resume=True, labels_path=None):
    """
    Extract features from left and right lung regions separately.
    
    Args:
        segmented_dir: Directory containing left and right lung segmentations
        original_images_dir: Directory containing original images
        output_base_dir: Base output directory 
        params: PyRadiomics parameters
        batch_size: Batch size for processing
        resume: Whether to resume from existing results
        labels_path: Optional path to labels CSV file
    """


    os.makedirs(output_base_dir, exist_ok=True)


    images_csv_path = os.path.join(output_base_dir, 'images.csv')
    images_df = create_images_csv_from_lung_segmentation(segmented_dir, original_images_dir, images_csv_path)

    if images_df is None:
        print("Error: Could not create images.csv file")
        return None


    lung_regions = {
        "left_lung": {
            "name": "Left Lung",
            "label": 1,
            "output_dir": "/home/akp4895/DeepLearning/fusion_multimodal/radiomics/radiomics_output_seven_experts/lobe_features/left_lung"
        },
        "right_lung": {
            "name": "Right Lung",
            "label": 2,
            "output_dir": "/home/akp4895/DeepLearning/fusion_multimodal/radiomics/radiomics_output_seven_experts/lobe_features/right_lung"
        }
    }

    all_lung_features = []


    for region_key, region_info in lung_regions.items():
        print(f"\nProcessing {region_info['name']} (label {region_info['label']})")


        lung_params = params.copy()
        if 'setting' not in lung_params:
            lung_params['setting'] = {}
        else:
            lung_params['setting'] = lung_params['setting'].copy()


        lung_params['setting']['label'] = region_info['label']


        lung_output_dir = region_info['output_dir']
        os.makedirs(lung_output_dir, exist_ok=True)


        lung_features = extract_features(
            images_csv_path,
            lung_output_dir,
            lung_params,
            batch_size=batch_size,
            resume=resume,
            label=region_info['label']
        )

        if lung_features is not None:

            lung_features['Region'] = region_info['name']
            lung_features['RegionType'] = 'Lung'
            lung_features['LungLabel'] = region_info['label']


            lung_csv_path = os.path.join(lung_output_dir, f'{region_key}_features.csv')
            lung_features.to_csv(lung_csv_path, index=False)
            print(f"Saved features for {region_info['name']} to {lung_csv_path}")

            all_lung_features.append(lung_features)


    if all_lung_features:
        combined_features = pd.concat(all_lung_features, ignore_index=True)


        combined_dir = os.path.join(output_base_dir, 'combined_lung_features')
        os.makedirs(combined_dir, exist_ok=True)
        combined_path = os.path.join(combined_dir, 'left_right_lung_features.csv')
        combined_features.to_csv(combined_path, index=False)
        print(f"Saved combined features (left + right lung) to {combined_path}")


        if labels_path and os.path.exists(labels_path):
            merge_features_with_labels(combined_path, labels_path, combined_dir)

        return combined_features
    else:
        print("No features were generated!")
        return None

def merge_features_with_labels(features_path, labels_path, output_dir):
    """Merge extracted features with labels"""
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    print(f"Loaded {len(features_df)} feature entries and {len(labels_df)} label entries")

    if 'Filename' not in features_df.columns or 'Filename' not in labels_df.columns:
        if 'ScanID' in labels_df.columns:
            labels_df['Filename'] = labels_df['ScanID'].apply(lambda x: f"{x}.nii.gz")

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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract radiomics features from left and right lung regions')
    parser.add_argument('--segmented_dir', type=str, required=True,
                        help='Path to directory containing left and right lung segmentations')
    parser.add_argument('--original_images_dir', type=str, required=True,
                        help='Path to directory containing original images')
    parser.add_argument('--output_dir', type=str, default='radiomics_output_lungs',
                        help='Base directory to save output files')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to PyRadiomics configuration YAML file')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels.csv file to merge with features')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of images to process in each batch')
    parser.add_argument('--no_resume', action='store_true',
                        help='Do not resume from previous extraction')
    parser.add_argument('--gpu', type=str, default="7",
                        help='GPU ID to use (default: 7)')

    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Set CUDA_VISIBLE_DEVICES to GPU {args.gpu}")


    params = create_pyradiomics_params(args.config)


    print("Extracting features from left and right lung regions")
    extract_features_left_right_lung(
        segmented_dir=args.segmented_dir,
        original_images_dir=args.original_images_dir,
        output_base_dir=args.output_dir,
        params=params,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        labels_path=args.labels
    )
