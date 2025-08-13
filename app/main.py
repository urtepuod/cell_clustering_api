import streamlit as st
import os
import tempfile
import zipfile
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ------------------------------------------------------------------

from app.processor import process_all_dirs
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None  # app will still run for manual uploads

st.set_page_config(page_title="Cell Counting App", layout="wide")
st.title("🔬 Cell Counting App")

model_choice = st.selectbox("Choose a segmentation model", ["temporal_5"])
clustering_method = st.selectbox("Choose clustering method", ["Distance-based", "HDBSCAN"])

# --- Upload OR Sample ---
uploaded_zip = st.file_uploader("Upload a ZIP of TIFF folders", type="zip")

# point to your dataset repo + filenames there
DATASET_ID = "urtepuod/cell_clustering_samples/sample_cells"   # <-- change to your dataset repo
SAMPLES = {
    "Non-motile cells demo": "24_03_02_05_34_27.zip",
    "Motile cells demo": "24_11_19_01_25_31.zip",
    # add more: "Label": "filename.zip"
}
sample_choice = st.selectbox("…or try a sample dataset", ["(none)"] + list(SAMPLES.keys()))

if st.button("Run Analysis", type="primary"):
    try:
        with st.spinner("Processing..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                # 1) decide where the zip comes from
                if uploaded_zip is not None:
                    zip_path = os.path.join(tmpdir, "uploaded.zip")
                    with open(zip_path, "wb") as f:
                        f.write(uploaded_zip.read())
                elif sample_choice != "(none)":
                    if hf_hub_download is None:
                        st.error("huggingface_hub not available; can’t fetch sample. Please upload a ZIP.")
                        st.stop()
                    zip_path = hf_hub_download(
                        repo_id=DATASET_ID,
                        repo_type="dataset",
                        filename=SAMPLES[sample_choice],
                    )
                else:
                    st.warning("Please upload a ZIP or select a sample dataset.")
                    st.stop()

                # 2) extract zip into tmpdir
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(tmpdir)

                # 3) run your pipeline
                output_path, scatter_path = process_all_dirs(tmpdir, model_choice, clustering_method)

                # 4) offer download + image
                with open(output_path, "rb") as f:
                    st.download_button("Download Cell Count Summary", f, file_name="cell_counts_summary.xlsx")

                if scatter_path and os.path.exists(scatter_path):
                    st.subheader("3D Cell Count Scatter Plot")
                    st.image(scatter_path, caption="3D object clustering", use_column_width=True)

        st.success("✅ Processing complete.")
    except Exception as e:
        st.exception(e)  # surface errors instead of a blank screen
