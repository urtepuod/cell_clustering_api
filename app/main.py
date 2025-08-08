import streamlit as st
import os
import tempfile
import zipfile
from app.processor import process_all_dirs

st.set_page_config(page_title="Cell Counting App", layout="wide")
st.title("🔬 Cell Counting App")

model_choice = st.selectbox("Choose a segmentation model", ["temporal_5"])
clustering_method = st.selectbox("Choose clustering method", ["Distance-based", "HDBSCAN"])

uploaded_zip = st.file_uploader("Upload a ZIP of TIFF folders", type="zip")

if uploaded_zip and st.button("Run Analysis"):
    with st.spinner("Processing..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            output_path, scatter_path = process_all_dirs(tmpdir, model_choice, clustering_method)

            with open(output_path, "rb") as f:
                st.download_button("Download Cell Count Summary", f, file_name="cell_counts_summary.xlsx")

            if scatter_path and os.path.exists(scatter_path):
                st.subheader("3D Cell Count Scatter Plot")
                st.image(scatter_path, caption="3D object clustering", use_container_width=True)

    st.success("✅ Processing complete.")
