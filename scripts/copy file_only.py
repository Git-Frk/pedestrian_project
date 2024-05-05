import os
import shutil
from tqdm import tqdm


def copy_files(src_dir, dst_dir):
    # Ensure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Iterate over files in the source directory
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)

        # Check if the current item is a file
        if os.path.isfile(src_file):
            # Copy the file to the destination directory
            shutil.copy(src_file, dst_dir)
        else:
            copy_files(src_file, dst_dir)


# Example usage:
source_directory = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/eurocity persons/model validation data/labels'
destination_directory = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/eurocity persons/ECP data/img-ann unpacked/labels'

copy_files(source_directory, destination_directory)
