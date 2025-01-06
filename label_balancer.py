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


DATASET_NAME = "flow_600_200"
SEED = 42

DEST_FOLDER = "balanced_data"

def DataBalancer(images_to_aumgment, label_dir): 
    """
    Balances a label by starting with the site id
    with the least amount of images and so forth
    """

    # Obtain image directory
    image_list = os.listdir(label_dir)

    # Declare dictionary to count number of images in site_id
    site_ids_count = {}

    # Declare dictionary to store the full_paths of
    # each image within a site id
    site_path = {}          # dictionary with a list inside
                            # e.g. [site_id: ]

    # Declare dictionary to store file names to name the files
    img_file_name = {}

    # Fetch date for naming convention
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%m%d%y")

    # Count site_id and append full_file_path
    total_files = 0
    for file_name in image_list:
        total_files += 1
        full_file_path = os.path.join(label_dir, file_name)
        image_name = os.path.basename(full_file_path)
        site_id = image_name.split("_")[0]

        if site_id in site_ids_count:
            site_ids_count[site_id] += 1
            site_path[site_id].append(full_file_path)
            img_file_name[site_id].append(image_name)
        else:
            site_ids_count[site_id] = 1
            site_path[site_id] = [full_file_path]
            img_file_name[site_id] = [image_name]

    # sort the dictionary, and turn it into a list
    sorted_list = SortDict(site_ids_count)
    
    CreateDir(new_folder_name)
    destination = os.path.join(root_dir, new_folder_name)

    # # Transfer all the images from the max to the
    # # new directory
    # max_site = sorted_list[-1][0]
    # for site_img in range(max):
    #     shutil.copy(site_path[max_site][site_img], destination)

    # # delete the site from all dict and list
    # del site_path[max_site]
    # del site_ids_count[max_site]
    # del img_file_name[max_site]
    # sorted_list.remove(sorted_list[-1])

    # # for both the shuffling between the site_path
    # # and img_file_name to be the same
    # seed = 42

    # # copy image_count amount of images from each site ids
    # # onto the new folder, and augment the rest to meet the
    # # max amount of images
    # site_number = 0
    # for site in site_path:
    #     # copy all the original images
    #     for site_img in range(site_ids_count[site]):
    #         shutil.copy(site_path[site][site_img], destination)
        
    #     # randomly shuffle the list contents
    #     # so that the balancing is not deterministic
    #     random.seed(seed)
    #     random.shuffle(site_path[site])
    #     random.shuffle(img_file_name[site])

    #     # already used all of the original images
    #     # ... now augment the original images,
    #     # to reach the max. The code is applying
    #     # the minimum degree of rotation to each
    #     # image in order to reach the image count
    #     image_count_aug = max - site_ids_count[site]    # --> number of images to augment to reach max
    #     counter = 1                               # --> counter for number of images currently augmented 
    #     switch = -1                               # --> (see more down), rotations only or horizontal flipped rotations
    #     theta = THETA
    #     fact = FACT

    #     switch_ended = 1                          # ---> after the switch is triggered two times, increase THETA and FACT
    #     while counter <= image_count_aug:            # --> continue to augment in case rotation or flip rotation is not enough
    #         #print(f"There are {image_count_aug} images to augment in site {site}, and we are in {counter}")
    #         for site_img in range(site_ids_count[site]):
    #             new_destination = os.path.join(destination, f"{img_file_name[site][site_img]}_augmented_{counter}.JPG")
    #             if counter > image_count_aug:       # --> if we reached enough images within the loop
    #                 #print(f"\nBroke when counter was {counter}, and image_count_aug was {image_count_aug}\n")
    #                 break
    #             elif switch == -1: #  case 1: only rotations
    #                 augmented_img = Data_augmentation(site_path[site][site_img], theta, fact, False)
    #                 cv.imwrite(new_destination, augmented_img)
    #                 counter+=1
    #             elif switch == 1:  # case 2: only horizontal flipped rotations
    #                 augmented_img = Data_augmentation(site_path[site][site_img], theta, fact, True)
    #                 cv.imwrite(new_destination, augmented_img)
    #                 counter+=1
    #         switch = switch*-1
    #         if switch_ended % 2 == 0:
    #             theta+=5
    #             fact+=0.3 # --> these THETA and FACT values ensure no black padding on final image
    #         switch_ended+=1
    #     site_number+=1
    #     #print(f"There are {len(site_path)+1} number of sites, and we are in {site_number}")

    # print(f"\n Data balanced is completed for {label}!\n")
    # print(f"\n Original dataset contained {total_files} number of images\n")
    # print(f"\n Augmented dataset contains {max * (len(sorted_list)+1)} number of images\n")
    return 

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

    all_label_dirs = [os.path.join(dataset_dir, "1"),
                        os.path.join(dataset_dir, "2"), 
                        os.path.join(dataset_dir, "3"), 
                        os.path.join(dataset_dir, "4"), 
                        os.path.join(dataset_dir, "5"), 
                        os.path.join(dataset_dir, "6")]   

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

    # # Create the destination directories
    # dest_dir_list = [os.path.join(dest_dir, "1"), 
    #                  os.path.join(dest_dir, "2"), 
    #                  os.path.join(dest_dir, "3"), 
    #                  os.path.join(dest_dir, "4"), 
    #                  os.path.join(dest_dir, "5"), 
    #                  os.path.join(dest_dir, "6")] 

    # # Copy all directories to the destination directories
    # max_workers = 10
    # with ThreadPoolExecutor(max_workers=max_workers) as executor: 
    #     for i in range(3): 
    #         executor.submit(shutil.copytree, source_dir, dest_dir_list[i])


    # # Copy all directories to the destination directories
    # for i in range(6): 
    #     shutil.copytree(source_dir, destination_dir)

    # if category_1_count > category_2_count: 
    #     print("\nCategory 1 larger than Category 2\n")
    # else: 
    #     print("\nCategory 2 larger than Category 1\n")

    #     images_to_augment = category_2_count - category_1_count
    #     print(f"\nImages to augment: {images_to_augment}\n")

    #     # Get counts of each label in category 1
    #     label_1_count = len(label_1_images)
    #     label_2_count = len(label_2_images)
    #     label_3_count = len(label_3_images)

    #     # Compute the complementary probabilities for sampling
    #     total_images_cat1 = label_1_count + label_2_count + label_3_count
    #     p_label_1 = 1 - (label_1_count / total_images_cat1)
    #     p_label_2 = 1 - (label_2_count / total_images_cat1)
    #     p_label_3 = 1 - (label_3_count / total_images_cat1)

    #     # Normalize the probabilities to 1, before sampling
    #     p_total = p_label_1 + p_label_2 + p_label_3
    #     p_label_1 /= p_total
    #     p_label_2 /= p_total
    #     p_label_3 /= p_total

    #     # Use numpy to sample which labels to augment
    #     sample = list(np.random.choice([1, 2, 3], images_to_augment, p=[p_label_1, p_label_2, p_label_3], replace=True))
        
    #     # Count how many images to augment for each label
    #     labels_to_augment_list = [sample.count(1), sample.count(2), sample.count(3)]
    #     all_label_dir = [label_1_dir, label_2_dir, label_3_dir]

    #     # 




    #     print(labels_to_augment_list)

    #     """ADD SUBPROCESS HERE"""
    #     """ALSO NEED TO COPY THE ORIGINAL DATA TO A DESTINATION FOLDER"""
    #     for i in range(len(labels_to_augment_list)):
    #         print(f"\nAugmenting label {i+1}\n")
    #         DataBalancer(labels_to_augment_list[i], all_label_dir[i])

if __name__ == "__main__":
    ImageToAugment()
    # print(f"\n-----Balancing/augmenting dataset-----\n")
    # for label in FOLDER_LABELS: 
    #     print(f"\nProcessing label {label}\n")
    #     DataBalancer(label, THETA, FACT)
    # print(f"\n-----Checking if the dataset is balanced-----\n")
    # for label in FOLDER_LABELS:
    #     Data_Is_Balanced(f"balanced_{label}")
    # print(f"\nYou indicated {ADD_DISCARDED} to add discarded data")
    # if ADD_DISCARDED: 
    #     for label in FOLDER_LABELS:
    #         AddDiscardedData(label)
    #         print(f"\nPlease check your directory for the new folder created with discarded data\nFeel free to merge the two datasets\n")
        
    














