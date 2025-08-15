import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="cellpose")

import streamlit as st
import os, sys, tempfile, zipfile

# Ensure we can import app.processor whether Spaces runs app.py or app/main.py
HERE = os.path.dirname(os.path.abspath(__file__))          # .../app
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))   # repo root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- import processor with diagnostics
try:
    from app.processor import process_all_dirs  # preferred
except ImportError:
    try:
        from processor import process_all_dirs  # fallback
    except ImportError as e:
        st.error("Couldn't import process_all_dirs from processor.py")
        st.code(str(e))
        try:
            st.write("Files in app/:", os.listdir(HERE))
        except Exception:
            pass
        st.stop()

# Optional Hub samples
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

st.set_page_config(page_title="Cell Counting App", layout="wide")
st.title("🔬 Cell Counting App")
st.caption("startup ok")

model_choice = st.selectbox("Choose a segmentation model", ["temporal_5"])
clustering_method = st.selectbox("Choose clustering method", ["Distance-based", "HDBSCAN"])

uploaded_zip = st.file_uploader("Upload a ZIP of TIFF folders", type="zip")

# Your dataset repo on the Hub (namespace/repo only)
DATASET_ID = "urtepuod/cell_clustering_samples"
SAMPLES = {
    "Motile cells demo": "sample_cells/24_03_02_05_34_27.zip",
    "Non-motile cells demo":     "sample_cells/24_11_19_01_25_31.zip",
}
sample_choice = st.selectbox("…or try a sample dataset", ["(none)"] + list(SAMPLES.keys()))

if st.button("Run Analysis", type="primary"):
    try:
        with st.spinner("Processing..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                # --- 1) Decide which ZIP to use
                st.info("step 1: choosing ZIP…")
                if uploaded_zip is not None:
                    zip_path = os.path.join(tmpdir, "uploaded.zip")
                    with open(zip_path, "wb") as f:
                        f.write(uploaded_zip.read())
                elif sample_choice != "(none)":
                    if hf_hub_download is None:
                        st.error("`huggingface_hub` not available; please upload a ZIP instead.")
                        st.stop()
                    zip_path = hf_hub_download(
                        repo_id=DATASET_ID,
                        repo_type="dataset",
                        filename=SAMPLES[sample_choice],
                    )
                else:
                    st.warning("Please upload a ZIP or select a sample dataset.")
                    st.stop()

                # --- 2) Extract into tmpdir
                st.info("step 2: extracting ZIP…")
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(tmpdir)
                except zipfile.BadZipFile:
                    st.error("Uploaded file is not a valid ZIP.")
                    st.stop()

                # --- 3) Run your pipeline
                st.info("step 3: running pipeline…")
                output_path, scatter_path = process_all_dirs(tmpdir, model_choice, clustering_method)

                # --- 4) Outputs
                st.info("step 4: preparing outputs…")
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
                    st.image(scatter_path, caption="3D object clustering", use_container_width=True)

        st.success("✅ Processing complete.")
    except Exception as e:
        st.error("An error occurred while processing:")
        st.exception(e)
