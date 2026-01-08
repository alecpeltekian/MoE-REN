import os
import subprocess

# Directory paths
source_dir = '/data2/akp4895/1mm_OrientedImages'
target_dir = '/data2/akp4895/1mm_OrientedImagesSegmented_Lungs'

# Set GPU environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def create_target_path(file_path):
    relative_path = os.path.relpath(file_path, source_dir)
    return os.path.join(target_dir, relative_path)

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.nii.gz'):
            source_file = os.path.join(root, file)
            target_file = create_target_path(source_file)
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            
            modelname = 'R231'
            command = f'lungmask {source_file} {target_file} --modelname {modelname}'
            
            print(f'Executing: {command}')
            subprocess.run(command, shell=True)