from cellpose import models
from skimage import io, measure, color, transform
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import re
import seaborn as sns
import hdbscan
import matplotlib; matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

# Parameters
MIN_CLUSTER_SIZE = 5  # min cluster size for a cluster to be considered valid
MAX_GAP = 15  # how many times a cell is to reoccur throughout the stack
SKIP_ALLOWED = 1  # cell to be counted as the same cell even if one segmentation mask is missing
MIN_SAMPLES = 3 # Set min_samples to 3 for HDBSCAN to cluster the same cell across frames

def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_working_model(path_to_model):
    # Load and Run Cellpose Model
    model = models.CellposeModel(pretrained_model=path_to_model, gpu=True)
    return model


def update_processed_directories_log(directory, log_file_path):
    """Update the log file with the processed directory."""
    with open(log_file_path, 'a') as log_file:
        log_file.write(directory + '\n')
    print(f"Directory {directory} added to log.")


def process_directory(directory, model, model_identifier):
    images = load_images(directory)

    # Create a directory for results inside the current directory
    results_dir = os.path.join(directory, f'results_model{model_identifier}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Run cellpose model
    channels = [[0, 0]]  # 0, 0 means grayscale image
    masks, flows, styles = model.eval(images, diameter=30, channels=channels, compute_masks=True)

    # Save Masks, Overlays, and create 3D Stack
    save_masks_overlay(images, masks, results_dir)
    create_3d_stack(masks, results_dir)

    # Count Objects and Save the Count
    number_of_cells = count_objects_hdbscan(masks, results_dir, z_scaling_factor=2.0, min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES)
    return number_of_cells



def load_images(path_to_images, image_resize_factor=3):
    img_dir = path_to_images
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.tiff')]
    sorted_image_files = sorted(image_files, key=numerical_sort_key)

    images = []
    for image_file in sorted_image_files:
        image_path = os.path.join(img_dir, image_file)
        image = io.imread(image_path)

        # Resize the image
        new_height = image.shape[0] // image_resize_factor
        new_width = image.shape[1] // image_resize_factor

        shrunken_image = transform.resize(image, (new_height, new_width), anti_aliasing=True)
        images.append(shrunken_image)
    return images


def save_masks_overlay(images, masks, save_directory):
    # Ensure the directories for saving masks and overlays exist
    masks_dir = os.path.join(save_directory, 'masks')
    overlay_dir = os.path.join(save_directory, 'overlay')
    Path(masks_dir).mkdir(parents=True, exist_ok=True)
    Path(overlay_dir).mkdir(parents=True, exist_ok=True)

    # Save masks and overlays
    for i, (image, mask) in enumerate(zip(images, masks)):
        mask_path = os.path.join(masks_dir, f'image_{i}.png')
        overlay_path = os.path.join(overlay_dir, f'image_{i}.png')

        plt.imsave(mask_path, mask)
        labeled_image = color.label2rgb(mask, image, alpha=0.2)
        plt.imsave(overlay_path, labeled_image)

    return [color.label2rgb(mask, image, alpha=0.2) for image, mask in zip(images, masks)]


def create_3d_stack(masks, save_directory):
    from pathlib import Path
    from skimage import measure
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    stack_dir = os.path.join(save_directory, "3d_images")
    Path(stack_dir).mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, mask in enumerate(masks):
        labeled_mask, num_labels = measure.label(mask, return_num=True)
        for label in range(1, num_labels + 1):
            object_indices = np.argwhere(labeled_mask == label)
            z_coords = np.full((object_indices.shape[0],), i)
            color = np.random.rand(3,).tolist()  # Convert numpy array to list
            ax.scatter(object_indices[:, 1], object_indices[:, 0], z_coords, color=color, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Slice (Object Index)')
    plt.savefig(os.path.join(stack_dir, "3d_stack.png"), dpi=300)

def split_same_slice_duplicates(labels, zs, next_new_label):
    """
    Ensure a cluster label never appears more than once in the same Z-slice.
    Returns the corrected label array and the next available new-label id.
    """
    labels = labels.copy()
    for z in np.unique(zs):
        idx = np.where(zs == z)[0]          # indices of points in slice z
        slice_labels = labels[idx]
        counts = Counter(slice_labels)
        for lbl, cnt in counts.items():
            if lbl == -1 or cnt == 1:
                continue                    # noise or unique → fine
            dup_idx = idx[slice_labels == lbl][1:]  # all but first
            for i in dup_idx:
                labels[i] = next_new_label  # re-assign
                next_new_label += 1
    return labels, next_new_label

def kill_same_slice_duplicates(labels, zs):
    seen = set()
    for i, (lbl, z) in enumerate(zip(labels, zs)):
        if lbl == -1:
            continue
        key = (lbl, z)
        if key in seen:
            labels[i] = -1          # duplicate → noise, NOT a new cluster
        else:
            seen.add(key)
    return labels


def count_objects_hdbscan(
        masks, save_directory,
        *, z_scaling_factor=2.0,
           min_cluster_size=MIN_CLUSTER_SIZE,
           min_samples=MIN_SAMPLES,
         debug=False):

    # 1️ collect centroids
    centers = []
    for z, mask in enumerate(masks):
        labeled, _ = measure.label(mask, return_num=True)
        for r in measure.regionprops(labeled):
            cy, cx = r.centroid
            centers.append((cx, cy, z * z_scaling_factor))
    if not centers:
        return 0
    centers = np.array(centers)
    if debug:                           # << diagnostic block
        xy = centers[:, :2]
        d  = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
        nn = np.partition(d + np.eye(len(d))*1e9, 1)[:, 1]
        print(f"median nn = {np.median(nn):.1f} px,   "
              f"95-perc = {np.percentile(nn,95):.1f} px")
    

    # 2 HDBSCAN in 3-D
    clusterer = hdbscan.HDBSCAN(
              min_cluster_size = min_cluster_size,
              min_samples      = min_samples,
              metric           = 'euclidean',
              cluster_selection_epsilon = 5)   
    labels = clusterer.fit_predict(centers[:, :2])     

    # 3️ split duplicates within each slice
    zs = (centers[:, 2] / z_scaling_factor).astype(int)   
    labels = kill_same_slice_duplicates(labels, zs)     # turn dup-in-slice → –1
    n_clusters= len(set(labels) - {-1})
    # 4️ (plotting + saving unchanged) …
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    palette = sns.color_palette("deep", max(n_clusters, 1))
    colours = [palette[l] if l >= 0 else (0, 0, 0) for l in labels]
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=colours, s=8)
    ax.text2D(0.05, 0.95, f"Cells detected: {n_clusters}", transform=ax.transAxes)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (scaled)")
    out_dir = Path(save_directory) / "3d_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "object_count_scatter_plot.png", dpi=300)
    plt.close(fig)

    # ----- 4 Persist plain-text count for your Excel aggregation step
    with open(Path(save_directory) / "cell_count.txt", "w") as fh:
        fh.write(str(n_clusters))
    
    return n_clusters


def save_results_to_excel(folder_names, cell_counts, output_path):
    """Save cell count results to an Excel file."""
    results_df = pd.DataFrame({'Folder Name': folder_names, 'Cell Count': cell_counts})
    results_df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")


def collect_and_save_counts(base_directory, model_identifier, output_file):
    folder_names = []
    cell_counts = []

    # Iterate through each directory in the base directory
    for dir_name in os.listdir(base_directory):
        dir_path = os.path.join(base_directory, dir_name)
        results_dir_name = f'results_model{model_identifier}'
        results_dir_path = os.path.join(dir_path, results_dir_name)
        count_file_path = os.path.join(results_dir_path, 'cell_count.txt')

        # Check if the cell count file exists
        if os.path.isdir(dir_path) and os.path.exists(count_file_path):
            with open(count_file_path, 'r') as count_file:
                count = count_file.read().strip()
                folder_names.append(dir_name)
                cell_counts.append(int(count))

    # Save the results to an Excel file
    if folder_names and cell_counts:
        results_df = pd.DataFrame({'Folder Name': folder_names, 'Cell Count': cell_counts})
        results_df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No results to save.")
