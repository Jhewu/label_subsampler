# Label Balancer (Overview)

Balances and performs the necessary image augmentation (e.g., 5-degree rotations + horizontal flip) to river stream images from **CT DEEP**, preparing the dataset for image classification training. This iteration focuses on the **two-category problem** (label 1, 2, 3 and label 4, 5, 6).

---

## The Augmentation Process

The default augmentations are:
- **Rotations** every 5 degrees (from 5° to 30°) 
- **Horizontal flip** before moving to the next degree rotation.

The script will augment the category with the least amount of images. After determining which category has fewer images, it will:
1. Augment the labels with fewer images.
2. Within each label, augment the `site_ids` with the least amount of images.

---

## Arguments

- `in_dir`: Directory containing the labeled folders (e.g., `1`, `2`, ... `6`).
- `out_dir` (default = current working directory).
- `theta` (default = 5): The angle of rotation. If you change `theta`, you must also adjust `fact`.
- `fact` (default = 1.3): The zoom factor after rotation.
- `multiplier` (default = 6): The number of times to rotate. If it's 6, it will rotate from 5° to 30° with horizontal flips in between, resulting in a final **13x multiplier** per image.

### Example Usage:

```bash
python3 script.py --in_dir "path_to_dataset"
