from cellpose import models
from cellpose.io import imread
from skimage import io, measure, color, segmentation, transform
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import re


def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split ('([0-9]+)', s)]
    
def load_working_model(path_to_model):
# Load and Run Cellpose Model
    model =  models.CellposeModel(pretrained_model=path_to_model, gpu=True)
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
    number_of_cells = count_objects(masks, results_dir)
    count_filename = os.path.join(results_dir, 'cell_count.txt')
    with open(count_filename, 'w') as file:
        file.write(str(number_of_cells))

    return number_of_cells


def load_images(path_to_images):
    img_dir = path_to_images
    # Sort the files using the updated numerical_sort_key
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.tiff')]
    sorted_image_files= sorted(image_files, key=numerical_sort_key)

    images = []
    for image_file in sorted_image_files:
        image_path = os.path.join(img_dir, image_file)
        image = io.imread(image_path)
        # Calculate the new dimensions
        new_height = image.shape[0] // 3
        new_width = image.shape[1] // 3

        # Resize the image
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



#Create 3d stack

def create_3d_stack(masks,save_directory):
    stack_dir = os.path.join(save_directory, "3d_images")
    Path(stack_dir).mkdir(parents=True, exist_ok=True)
    # List of mask images
    mask_images = masks  # List of your mask images

    # Create a 3D projection

    # Create a 3D-like scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Loop through the masks and create scatter points for object locations
    for i, mask in enumerate(mask_images):
        labeled_mask, num_labels = measure.label(mask, return_num=True)
        for label in range(1, num_labels + 1):
            object_indices = np.argwhere(labeled_mask == label)
            z_coords = np.full((object_indices.shape[0],), i)
            ax.scatter(object_indices[:, 1], object_indices[:, 0], z_coords, c=np.random.rand(3,), marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Slice (Object Index)')

    plt.savefig(os.path.join(stack_dir, "3d_image.png"), dpi=300)


def create_3d_projection(masks,save_directory):
    projection_dir=os.path.join(save_directory, '3d_projection')
    Path(projection_dir).mkdir(parents=True, exist_ok=True)

    # List of mask images
    mask_images = masks

    # Initialize a dictionary to store unique colors for labels
    label_to_color = {}

    # Loop through the masks and create scatter points for object locations
    for i in range(1, len(mask_images)):
        prev_mask = mask_images[i - 1]
        curr_mask = mask_images[i]
        
        # Label objects in the previous and current masks
        labeled_prev_mask, num_prev_labels = measure.label(prev_mask, return_num=True)
        labeled_curr_mask, num_curr_labels = measure.label(curr_mask, return_num=True)
        
        # Loop through objects in the current mask
        for label_curr in range(1, num_curr_labels + 1):
            object_curr = labeled_curr_mask == label_curr
            
            # Calculate overlap with objects in the previous mask
            overlaps = labeled_prev_mask * object_curr
            overlapping_labels = np.unique(overlaps)
            overlapping_labels = overlapping_labels[overlapping_labels != 0]
            
            # Loop through overlapping objects and check overlap ratio
            for label_prev in overlapping_labels:
                overlap_pixels = np.sum(overlaps == label_prev)
                total_pixels = np.sum(labeled_prev_mask == label_prev)
                overlap_ratio = overlap_pixels / total_pixels
                
                if overlap_ratio >= 0.4:
                    if label_prev not in label_to_color:
                        label_to_color[label_prev] = np.random.rand(3,)
                    
                    color = label_to_color[label_prev]
                    object_indices = np.argwhere(labeled_curr_mask == label_curr)
                    
                    plt.scatter(object_indices[:, 1], object_indices[:, 0], c=[color], marker='o')

        plt.xlabel('X')
        plt.ylabel('Y')

        plt.savefig(os.path.join(projection_dir, "3d_images/3d_projection.png"), dpi = 300)


def count_objects(masks, save_directory):
    # Initialize arrays to store center points
    centers = []

    # Load masks and determine centers
    for i, mask_file in enumerate(masks):
        mask = masks[i]
        # Label connected components in the mask
        labeled_mask = measure.label(mask > 0)
        
        # Calculate center of mass for each labeled region
        for region in measure.regionprops(labeled_mask):
            center_y, center_x = region.centroid
            centers.append([center_x, center_y, i])

    # Convert centers list to array
    centers = np.array(centers)

    # Define grouping distance- approximately 1um
    grouping_distance = 18.8

    # Initialize dictionary to store color assignments
    color_assignments = {}
    current_color = 0

    # Assign colors to center points based on grouping
    for center in centers:
        found_group = False
        for color, points in color_assignments.items():
            if any(np.linalg.norm(center[:2] - point[:2]) <= grouping_distance for point in points):
                color_assignments[color].append(center)
                found_group = True
                break
        if not found_group:
            color_assignments[current_color] = [center]
            current_color += 1

    # Create an XYZ scatter plot with unique colors for each group
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for color, points in color_assignments.items():
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=f'Group {color}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    total_objects = len(ax.legend().legend_handles)
    ax.legend([f'Number of total objects:{total_objects}'])

    # Save the scatter plot
    scatter_plot_path = os.path.join(save_directory, "3d_images/object_count_scatter_plot.png")
    plt.savefig(scatter_plot_path, dpi=300)

    # Save the total count to a text file
    total_objects = len(color_assignments)
    count_filename = os.path.join(save_directory, 'cell_count.txt')
    with open(count_filename, 'w') as count_file:
        count_file.write(str(total_objects))

    return total_objects


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
