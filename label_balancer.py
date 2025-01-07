"""
Data Balancer, balances the river image data. It compares the
site with the minimum amount of images (times a multiplier),
to the site with the maximum amount of images. If the multiplier,
does not reach, then the site is ignore. For all of the others that do
meet, apply the data augmentation required to meet the number of images
in the max. We are sacrificing some diversity, in order to get the maximum
amount of data, since diffusion models work well with a lot of data,
to generalize better.

The default augmentations are rotations every 5 degrees, from 5-30,
and then do horizontal flip, and then do 5-30 again. Therefore, we have
a total of 13x multiplier per site ID.

If you want to change the angle of rotations, you can change THETA, but you will 
need to recalculate the factor for upsampling (how much you will need to zoom into
the new image to crop out the padding). 

If you want to change the range of rotations, change MULTIPLIER. If you're doing
5-30 rotations, then it's 6x (from degree 5 to degree 10, there are 6 iterations). 
DO NOT CHANGE TOTAL_MULTIPLIER as it accounts for the original image and the horizontal flip. 
"""

# Import external libraries
import os
import shutil
import random
import cv2 as cv
import numpy as np
import datetime
from concurrent.futures import ThreadPoolExecutor

# Import script to check if data is balanced
from data_is_balanced import Data_Is_Balanced
from add_discarded_data import AddDiscardedData
from utils import *

"""CHANGE HYPERPARAMETERS"""
THETA = 5                                       # --> degree of rotations (per augmentation)   
FACT = 1.3                                      # --> multiplication factor to upsample (e.g. zoom into the image)
                                                # If you're using 5 degrees interval, then it takes 1.3x zoom factor on each 
                                                # image to crop out the padding resulted from rotation

                                                # If you change THETA, you must change FACT, recommended
                                                # to keep its default value


MULTIPLIER = 6                                  # If rotations from degree 5-30 (there are 6 iterations).
                                                # this number represents how many times, we need to iterate
                                                # to reach the final rotation degree

TOTAL_MULTIPLIER = (MULTIPLIER*2) + 1           # DO NOT CHANGE THIS VALUE

FOLDER_LABELS = ["6"]                            # the name of the folder
ADD_DISCARDED = True


# DATASET_NAME = "flow_600_200"
DATASET_NAME = "toy_dataset"
SEED = 42

DEST_FOLDER = "balanced_data"

def DataBalancer(images_to_aumgment, label_dir, dest_dir, label): 
    """
    Balances a label by starting with the site id
    with the least amount of images and so forth
    """
    print(images_to_aumgment)

    # Create the list of images
    image_list = os.listdir(label_dir)

    # Declare dictionary to count number of images in site_id
    site_ids_count = {}

    # Declare dictionary to store the full_paths of
    # each image within a site id
    site_path = {}          # dictionary with a list inside
                            # e.g. [site_id: 23]

    # Count site_id and append full_file_path
    total_files = 0
    for file_name in image_list:
        total_files += 1
        full_file_path = os.path.join(label_dir, file_name)
        site_id = file_name.split("_")[0]

        if site_id in site_ids_count:
            site_ids_count[site_id] += 1
            site_path[site_id].append(full_file_path)
            # img_file_name[site_id].append(image_name)
        else:
            site_ids_count[site_id] = 1
            site_path[site_id] = [full_file_path]
            # img_file_name[site_id] = [image_name]

    # Sort the dictionary, and turn it into a list
    sorted_list = SortDict(site_ids_count)    

    # Fetch date for naming convention
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%m%d%y")

    # Augment from the site ids with the lowest number of 
    # images until it reaches images_to_augment (total number to augment)
    counter = 1             # --> counter for number of images currently augmented 
    for element in sorted_list: 
        # Element is shaped like this: ('site_id', 2)
        # first element is the site_id (a string)
        # second element is the count

        site_index = 0
        for site in site_path[element[0]]: 
            # Now augment the original images,
            # to reach the max. The code is applying
            # the minimum degree of rotation to each
            # image in order to reach the image count
            switch = -1                                     # --> (see more down), rotations only or horizontal flipped rotations
            theta = THETA
            fact = FACT

            # After the switch is triggered two times, 
            # increase THETA and FACT            
            switch_ended = 1    

            # Continue to augment until you reached: 
            # 1) the end of images_to_augment
            # 2) the end of the multiplier (e.g. ran out of rotations and horizontal flip)
            for i in range(MULTIPLIER*2):
                basename = os.path.basename(site)
                new_file_name = f"{basename.split('_')[0]}_D{formatted_date}_{formatted_date}_{site_index}_{i}_ROT_AUG_{label}.JPG"
                new_destination = os.path.join(dest_dir, new_file_name)

                if counter > images_to_aumgment:       # --> if we reached enough images within the loop
                    print(f"\nReached enough images, balancing is finished for {label}")
                    return
                elif switch == -1: #  case 1: only rotations
                    augmented_img = Data_augmentation(site, theta, fact, False)
                    cv.imwrite(new_destination, augmented_img)
                    counter+=1
                elif switch == 1:  # case 2: only horizontal flipped rotations
                    augmented_img = Data_augmentation(site, theta, fact, True)
                    cv.imwrite(new_destination, augmented_img)
                    counter+=1

                switch = switch*-1
                if switch_ended % 2 == 0:
                    theta+=5
                    fact+=0.3   # --> these THETA and FACT values ensure no black padding on final image
                switch_ended+=1
            site_index+=1

def ImageToAugment():
    """
    This function will determine which label images to 
    augment based on the number of images in the two 
    category problem (label 1,2,3) and (label, 4, 5, 6)
    """

    """MAKE THIS CODE MORE READABLE"""
    # Obtain image directory
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, DATASET_NAME)
    dest_dir = os.path.join(root_dir, DEST_FOLDER)

    # Initialize the lists
    category_1_images = [] 
    category_2_images = []

    all_label_dirs = [os.path.join(dataset_dir, str(i)) for i in range(1, 7)]
    # all_label_dirs = [os.path.join(dataset_dir, "1"),
    #                     os.path.join(dataset_dir, "2"), 
    #                     os.path.join(dataset_dir, "3"), 
    #                     os.path.join(dataset_dir, "4"), 
    #                     os.path.join(dataset_dir, "5"), 
    #                     os.path.join(dataset_dir, "6")]   

    # Prepare category 1
    category_1_list = [
        os.listdir(all_label_dirs[0]), 
        os.listdir(all_label_dirs[1]), 
        os.listdir(all_label_dirs[2])]
    
    for i in range(len(category_1_list)): 
        category_1_images.extend(category_1_list[i])
    
    category_1_count = len(category_1_images)

    # Prepare category 2
    category_2_list = [
        os.listdir(all_label_dirs[3]), 
        os.listdir(all_label_dirs[4]), 
        os.listdir(all_label_dirs[5])]

    for i in range(len(category_2_list)): 
        category_2_images.extend(category_2_list[i])
    
    category_2_count = len(category_2_images)

    print(category_2_count)

    # Create the destination directories
    dest_label_dirs = [os.path.join(dest_dir, str(i)) for i in range(1, 7)]
    # dest_label_dirs = [os.path.join(dest_dir, "1"), 
    #                  os.path.join(dest_dir, "2"), 
    #                  os.path.join(dest_dir, "3"), 
    #                  os.path.join(dest_dir, "4"), 
    #                  os.path.join(dest_dir, "5"), 
    #                  os.path.join(dest_dir, "6")] 

    # Copy all directories to the destination directories

    if category_1_count > category_2_count: 
        print("\nCategory 1 larger than Category 2\n")

        images_to_augment = category_1_count - category_2_count
        print(f"Images to augment: {images_to_augment}\n")

        if category_2_count * TOTAL_MULTIPLIER < images_to_augment: 
            print("Category 2 does not contain enough images to augment to Category 1")
            print("The script will terminate...\n")
            return
        
        print("Category 2 does contain enough images to augment to Category 1")
        print("Proceeding to augment...\n")

        print(f"Copying original files to {dest_dir}...\n")
        max_workers = 10
        with ThreadPoolExecutor(max_workers=max_workers) as executor: 
            executor.map(copy_dir, all_label_dirs, dest_label_dirs)

        # Get counts of each label in category 1
        category_1_count_split = [
            len(category_1_list[0]), 
            len(category_1_list[1]), 
            len(category_1_list[2])]
        
        # Compute the complementary probabilities for sampling
        p_label_1 = 1 - (category_1_count_split[0] / category_1_count)
        p_label_2 = 1 - (category_1_count_split[1] / category_1_count)
        p_label_3 = 1 - (category_1_count_split[2] / category_1_count)

        # Normalize the probabilities to 1, before sampling
        p_total = p_label_1 + p_label_2 + p_label_3
        p_label_1 /= p_total
        p_label_2 /= p_total
        p_label_3 /= p_total

        print(p_label_1, p_label_2, p_label_3)

        # Use numpy to sample which labels to augment
        sample = list(np.random.choice([1, 2, 3], images_to_augment, p=[p_label_1, p_label_2, p_label_3], replace=True))
        
        # Count how many images to augment for each label
        images_to_aug_per_label = [sample.count(1), sample.count(2), sample.count(3)]

        print(images_to_aug_per_label)

        print(images_to_aug_per_label[0] + category_1_count_split[0])
        print(images_to_aug_per_label[1] + category_1_count_split[1])
        print(images_to_aug_per_label[2] + category_1_count_split[2])

        max_workers = 10
        with ThreadPoolExecutor(max_workers=max_workers) as executor: 
            for i in range(len(images_to_aug_per_label)):
                print(f"Augmenting label {i+1}...\n")
                executor.submit(DataBalancer, images_to_aug_per_label[i], 
                                all_label_dirs[i],
                                dest_label_dirs[i], 
                                f"L{i+1}")

    else: 
        print("\nCategory 2 larger than Category 1\n")

        images_to_augment = category_2_count - category_1_count
        print(f"Images to augment: {images_to_augment}\n")

        if category_1_count * TOTAL_MULTIPLIER < images_to_augment: 
            print("Category 1 does not contain enough images to augment to Category 2")
            print("The script will terminate...\n")
            return
        
        print("Category 1 does contain enough images to augment to Category 2")
        print("Proceeding to augment...\n")

        print(f"Copying original files to {dest_dir}...\n")
        max_workers = 10
        with ThreadPoolExecutor(max_workers=max_workers) as executor: 
            executor.map(copy_dir, all_label_dirs, dest_label_dirs)

        # Get counts of each label in category 1
        category_1_count_split = [
            len(category_1_list[0]), 
            len(category_1_list[1]), 
            len(category_1_list[2])]
        
        # Compute the complementary probabilities for sampling
        p_label_1 = 1 - (category_1_count_split[0] / category_1_count)
        p_label_2 = 1 - (category_1_count_split[1] / category_1_count)
        p_label_3 = 1 - (category_1_count_split[2] / category_1_count)

        # Normalize the probabilities to 1, before sampling
        p_total = p_label_1 + p_label_2 + p_label_3
        p_label_1 /= p_total
        p_label_2 /= p_total
        p_label_3 /= p_total

        print(p_label_1, p_label_2, p_label_3)

        # Use numpy to sample which labels to augment
        sample = list(np.random.choice([1, 2, 3], images_to_augment, p=[p_label_1, p_label_2, p_label_3], replace=True))
        
        # Count how many images to augment for each label
        images_to_aug_per_label = [sample.count(1), sample.count(2), sample.count(3)]

        print(images_to_aug_per_label)

        print(images_to_aug_per_label[0] + category_1_count_split[0])
        print(images_to_aug_per_label[1] + category_1_count_split[1])
        print(images_to_aug_per_label[2] + category_1_count_split[2])

        max_workers = 10
        with ThreadPoolExecutor(max_workers=max_workers) as executor: 
            for i in range(len(images_to_aug_per_label)):
                print(f"Augmenting label {i+1}...\n")
                executor.submit(DataBalancer, images_to_aug_per_label[i], 
                                all_label_dirs[i],
                                dest_label_dirs[i], 
                                f"L{i+1}")

if __name__ == "__main__":
    ImageToAugment()
    print("\nFinish augmenting the label")














