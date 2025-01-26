# Import external libraries
import os
import shutil

# Import helper functions
from utils import *

"""GLOBAL PARAMETERS"""

INPUT_DIR = "3"
OUTPUT_DIR = "L3"
SAMPLE_FROM_SITE = 400

def LabelSubsampler(): 
    # Obtain image directory
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, INPUT_DIR)
    dest_dir = os.path.join(root_dir, OUTPUT_DIR)

    # Create the destination folder
    CreateDir(dest_dir)

    # Obtain the list of the images
    image_list = os.listdir(dataset_dir)

    # Declare dictionary to count site_id
    site_ids_count = {}

    # Declare dictionary to store the full_paths of
    # each image within a site id
    site_path = {}      # dictionary with a list inside
                        # e.g. [site_id: 23]

    # Count site_id and append full_file_path
    total_files = 0
    for file_name in image_list:
        total_files += 1
        
        # Obtain full file path
        full_file_path = os.path.join(INPUT_DIR, file_name)

        # Obtain the site_id
        site_id = file_name.split("_")[0]
    
        if site_id in site_ids_count:
            # Case 1: if it's already in the list
            site_ids_count[site_id] += 1
            site_path[site_id].append(full_file_path)
        else:
            # Case 2: if it's not in the list
            site_ids_count[site_id] = 1
            site_path[site_id] = [full_file_path]

    # Sort the dictionary, and turn it into a list
    sorted_list = SortDict(site_ids_count)
    
    # This loop is iterating through each site_id
    for element in sorted_list: 
        # Element is shaped like this: ('site_id', 2)
        # first element is the site_id (a string)
        # second element is the count

        # Defined the site id
        site = element[0]

        # Define the counter to ensure certain number 
        # of sample per site
        counter = 0
        
        # This loop is iterating through each site image 
        # within the site_path
        for i in range(len(site_path[site])): 
            # Increase the counter
            counter+=1
            print(counter)

            # Get the image path
            image_path = site_path[site][i]

            # Obtain the basename of image_path
            # and Create the new destination
            basename = os.path.basename(image_path)
            new_dest = os.path.join(dest_dir, basename)
            
            shutil.copy(image_path, new_dest)             
            
            if SAMPLE_FROM_SITE == counter: 
                break
                
    print(f"\nThis is sorted list {sorted_list}\n")
    print(f"This is the length of site IDs {len(sorted_list)}")
    

if __name__ == "__main__":
    LabelSubsampler()
    