import os
from app import support_distance, support_hdbscan

def process_all_dirs(base_dir, model_identifier, method):
    # Choose the support module based on the clustering method
    if method == "HDBSCAN":
        sf = support_hdbscan
    else:
        sf = support_distance

    model_path = f"models/{model_identifier}"
    model = sf.load_working_model(model_path)

    folder_names = []
    cell_counts = []
    scatter_plot_path = None

    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path):
            print(f"Processing {dir_name}")
            count = sf.process_directory(dir_path, model, model_identifier)
            folder_names.append(dir_name)
            cell_counts.append(count)

            if scatter_plot_path is None:
                results_dir = os.path.join(dir_path, f'results_model{model_identifier}', "3d_images")
                scatter_path = os.path.join(results_dir, "object_count_scatter_plot.png")
                if os.path.exists(scatter_path):
                    scatter_plot_path = scatter_path

    output_file = os.path.join(base_dir, "cell_counts_summary.xlsx")
    sf.save_results_to_excel(folder_names, cell_counts, output_file)
    return output_file, scatter_plot_path
