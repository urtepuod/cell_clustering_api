import streamlit as st
import os
import sys
import tempfile
import zipfile

# --- ensure repo root is on sys.path so "app.*" works when running app/main.py directly
HERE = os.path.dirname(os.path.abspath(__file__))          # .../app
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))   # repo root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- import processor with useful diagnostics
try:
    from app.processor import process_all_dirs          # preferred
except ImportError as e1:
    try:
        # fallback if Python is treating this as a flat script and app/ is already on path
        from processor import process_all_dirs
    except ImportError as e2:
        st.error("Couldn't import process_all_dirs from processor.py")
        st.code(
            "ImportErrors:\n"
            f"  app.processor -> {e1}\n"
            f"  processor      -> {e2}\n\n"
            f"HERE={HERE}\nPROJECT_ROOT={PROJECT_ROOT}\n"
            f"sys.path[0:3]={sys.path[:3]} ..."
        )
        # show what's actually in the app folder to catch typos/missing files
        try:
            st.write("Files in app/:", os.listdir(HERE))
        except Exception:
            pass
        st.stop()
except Exception as e:
    # processor.py was found but crashed during import — show the real error
    st.error("processor.py raised an error during import:")
    st.exception(e)
    st.stop()
# Optional: fetch sample ZIPs from the Hub if huggingface_hub is available
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None  # app still works with manual uploads

st.set_page_config(page_title="Cell Counting App", layout="wide")
st.title("🔬 Cell Counting App")

model_choice = st.selectbox("Choose a segmentation model", ["temporal_5"])
clustering_method = st.selectbox("Choose clustering method", ["Distance-based", "HDBSCAN"])

# --- Upload OR Sample ---
uploaded_zip = st.file_uploader("Upload a ZIP of TIFF folders", type="zip")

# Your dataset repo on the Hub (NO subfolder here)
DATASET_ID = "urtepuod/cell_clustering_samples"

# Map labels to file paths inside the dataset repo
SAMPLES = {
    "Non-motile cells demo": "sample_cells/24_03_02_05_34_27.zip",
    "Motile cells demo":     "sample_cells/24_11_19_01_25_31.zip",
}

sample_choice = st.selectbox("…or try a sample dataset", ["(none)"] + list(SAMPLES.keys()))

if st.button("Run Analysis", type="primary"):
    try:
        with st.spinner("Processing..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                # 1) Decide which ZIP to use
                if uploaded_zip is not None:
                    zip_to_use = os.path.join(tmpdir, "uploaded.zip")
                    with open(zip_to_use, "wb") as f:
                        f.write(uploaded_zip.read())
                elif sample_choice != "(none)":
                    if hf_hub_download is None:
                        st.error("`huggingface_hub` not available; please upload a ZIP instead.")
                        st.stop()
                    # Download the selected sample file
                    zip_to_use = hf_hub_download(
                        repo_id=DATASET_ID,
                        repo_type="dataset",
                        filename=SAMPLES[sample_choice],  # subfolder path lives here
                    )
                else:
                    st.warning("Please upload a ZIP or select a sample dataset.")
                    st.stop()

                # 2) Extract ZIP
                try:
                    with zipfile.ZipFile(zip_to_use, "r") as zf:
                        zf.extractall(tmpdir)
                except zipfile.BadZipFile:
                    st.error("Uploaded file is not a valid ZIP.")
                    st.stop()

                # 3) Run the pipeline
                output_path, scatter_path = process_all_dirs(tmpdir, model_choice, clustering_method)

                # 4) Download results + show image (if produced)
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "Download Cell Count Summary",
                            f,
                            file_name="cell_counts_summary.xlsx",
                        )
                else:
                    st.warning("No results file was produced.")

                if scatter_path and os.path.exists(scatter_path):
                    st.subheader("3D Cell Count Scatter Plot")
                    st.image(scatter_path, caption="3D object clustering", use_column_width=True)

        st.success("✅ Processing complete.")
    except Exception as e:
        # Surface any unexpected errors instead of blank screen
        st.error("An error occurred while processing:")
        st.exception(e)
