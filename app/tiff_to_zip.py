import os
import zipfile

def zip_tiff_folders(base_dir, output_dir=None):
    if output_dir is None:
        output_dir = base_dir

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Skipping non-folder: {folder_name}")
            continue

        tiff_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]
        if not tiff_files:
            print(f"No TIFFs found in: {folder_name}")
            continue

        zip_name = os.path.join(output_dir, f"{folder_name}.zip")
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in tiff_files:
                file_path = os.path.join(folder_path, file)
                arcname = os.path.join(folder_name, file)
                zipf.write(file_path, arcname)
        print(f"Zipped: {zip_name}")

# Example usage
if __name__ == "__main__":
    base_dir = "/home/urte/device_data/raw/mg1655"
    zip_tiff_folders(base_dir)
