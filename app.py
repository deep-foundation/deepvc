import streamlit as st
import pandas as pd
from pathlib import Path
from deepcore import sort_duoblet
from deepvisual import visualize_link_doublet
from deepcore import cluster_links
import matplotlib.pyplot as plt
import io
from deepvisual import visualize_link_doublet_cluster

# --- set page config ---
st.set_page_config(
    page_title="DeepVC",
    layout="wide",
    page_icon="img/logo/logo.png"
)

# --- inject custom CSS ---
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- navigation control ---
# define pages in app
PAGES = {
    "Home": "Home",
    "Uploading data": "Uploading data",
    "Sorting the data": "Sorting the data",
    "Visualization of links": "Visualization of links",
    "Visualizing link clustering": "Visualizing link clustering",
    "Link clustering": "Link clustering"
}

# initialize session state for page navigation and file info
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'file_info' not in st.session_state:
    st.session_state.file_info = None

# function to change page
def change_page(page):
    st.session_state.current_page = page

# --- sidebar Navigation ---
st.sidebar.markdown("""
<style>
    .fake-title {
        font-size: 24px;
        font-weight: bold;
        padding: 10px 10px 5px 10px;
        text-align: center;
        color: #000000;
        cursor: pointer;
        display: block;
        margin-bottom: 20px;
    }
    .fake-title:hover {
        color: #3d3d3d;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.sidebar.columns([1,6,1])
with col2:
    if st.button("DeepVC", key="main_title"):
        change_page("Home")
    st.markdown("""
    <style>
        div[data-testid="stHorizontalBlock"] button {
            background: none !important;
            border: none !important;
            box-shadow: none !important;
            font-size: 24px !important;
            font-weight: bold !important;
            padding: 10px !important;
            color: #000000 !important;
            width: 100% !important;
        }
        div[data-testid="stHorizontalBlock"] button:hover {
            color: #3d3d3d !important;
            background: none !important;
        }
    </style>
    """, unsafe_allow_html=True)


# navigation links
st.sidebar.markdown("""
<div class="nav-section">Uploading</div>
""", unsafe_allow_html=True)
if st.sidebar.button("Uploading data", key="upload_btn"):
    change_page("Uploading data")

st.sidebar.markdown("""
<div class="nav-section">Doublet</div>
""", unsafe_allow_html=True)
if st.sidebar.button("Sorting the data", key="sort_btn"):
    change_page("Sorting the data")
if st.sidebar.button("Visualization of links", key="vis_btn"):
    change_page("Visualization of links")
if st.sidebar.button("Visualizing link clustering", key="vis_cluster_btn"):  # Added new button
    change_page("Visualizing link clustering")
if st.sidebar.button("Link clustering", key="cluster_btn"):
    change_page("Link clustering")

st.sidebar.markdown("""
<div class="nav-section">Triplet</div>
""", unsafe_allow_html=True)

# --- shared state via session_state ---
if "dataframe_buffer" not in st.session_state:
    st.session_state.dataframe_buffer = None
if "uploaded_file_info" not in st.session_state:
    st.session_state.uploaded_file_info = None

# --- page Content ---
# home Page
if st.session_state.current_page == "Home":
    st.markdown("""
    <style>
        .home-content {
            max-width: 800px;
            margin: 40px auto;
            text-align: center;
            color: #F5F5F5;
            font-size: 16px;
            line-height: 1.5;
        }
        .github-icon {
            margin-top: 50px;
            transition: transform 0.2s;
        }
        .github-icon:hover {
            transform: scale(1.1);
        }
    </style>
    
    <div class="home-content">
        <p>GUI methods from the DeepCore and DeepVisual libraries. DeepVC (Deep Visual Core) is distributed under the license of The Unlicense, developed by Deep.Foundation and serves as an intuitive tool for data management and visualization</p>
        <a href="https://github.com/deep-foundation" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/a/ae/Github-desktop-logo-symbol.svg" width="48" class="github-icon"/>
        </a>
    </div>
    """, unsafe_allow_html=True)

# upload page
elif st.session_state.current_page == "Uploading data":
    st.markdown("<h1 class='page-title'>Upload your CSV file</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.dataframe_buffer = df.to_csv(index=False)
            st.session_state.uploaded_file_info = {
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "rows": len(df)  # Сохраняем количество строк в session_state
            }
            st.success("File loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    if st.session_state.uploaded_file_info:
        st.markdown(f"""
        <div class="upload-info-box">
            <strong>Uploaded file:</strong> {st.session_state.uploaded_file_info['name']}<br>
            <strong>Size:</strong> {st.session_state.uploaded_file_info['size']} bytes<br>
            <strong>Rows:</strong> {st.session_state.uploaded_file_info.get('rows', 'N/A')}
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Используем dataframe_buffer вместо переменной df
            if st.session_state.dataframe_buffer:
                df_preview = pd.read_csv(io.StringIO(st.session_state.dataframe_buffer))
                st.write("Preview (first 5 rows):")
                st.dataframe(df_preview.head())
        except Exception as e:
            st.error(f"Error displaying data: {e}")

# sorting Page
elif st.session_state.current_page == "Sorting the data":
    st.markdown("<h1 class='page-title'>Sorting the Data</h1>", unsafe_allow_html=True)

    if st.session_state.dataframe_buffer is None:
        st.warning("Please upload data first.")
    else:
        df = pd.read_csv(io.StringIO(st.session_state.dataframe_buffer))

        if st.button("Process Data"):
            try:
                sorted_df = sort_duoblet(df)
                st.session_state.dataframe_buffer = sorted_df.to_csv(index=False)
                st.success("Data processed successfully!")
                st.dataframe(sorted_df.head())
            except Exception as e:
                st.error(f"Error processing data: {e}")

        st.download_button(
            "Download Processed CSV",
            data=st.session_state.dataframe_buffer,
            file_name="processed.csv",
            mime="text/csv"
        )

# visualization page
elif st.session_state.current_page == "Visualization of links":
    st.markdown("<h1 class='page-title'>Visualization of Links</h1>", unsafe_allow_html=True)

    if st.session_state.dataframe_buffer is None:
        st.warning("Please upload and process data first.")
    else:
        df = pd.read_csv(io.StringIO(st.session_state.dataframe_buffer))

        loop_color = st.color_picker("Loop Color", "#FFA500")
        edge_color = st.color_picker("Edge Color", "#000000")
        inter_edge_color = st.color_picker("Inter Edge Color", "#0000FF")
        background_color = st.color_picker("Background Color", "#FFFFFF")
        title = st.text_input("Visualization Title", "")
        color_title = st.color_picker("Title Color", "#000000")

        viz_placeholder = st.empty()
        download_placeholder = st.empty()

        if st.button("Generate Visualization"):
            try:
                plt.figure(figsize=(10, 8))
                
                visualize_link_doublet(
                    df,
                    loop_color=loop_color,
                    edge_color=edge_color,
                    inter_edge_color=inter_edge_color,
                    background_color=background_color,
                    title=title,
                    color_title=color_title
                )
                
                fig = plt.gcf()
                
                viz_placeholder.pyplot(fig)
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                buf.seek(0)
                
                download_placeholder.download_button(
                    "Download PNG",
                    data=buf,
                    file_name="visualization.png",
                    mime="image/png"
                )
                
                plt.close(fig)
                
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")

# visualizing link clustering page
elif st.session_state.current_page == "Visualizing link clustering":
    st.markdown("<h1 class='page-title'>Visualizing Link Clustering</h1>", unsafe_allow_html=True)

    if st.session_state.dataframe_buffer is None:
        st.warning("Please upload and process data first.")
    else:
        df = pd.read_csv(io.StringIO(st.session_state.dataframe_buffer))

        background_color = st.color_picker("Background Color", "#FFFFFF")
        color_title = st.color_picker("Title Color", "#000000")
        title = st.text_input("Visualization Title", "")
        
        viz_placeholder = st.empty()
        download_placeholder = st.empty()

        if st.button("Generate Clustering Visualization"):
            try:
                plt.figure(figsize=(12, 8))
                
                visualize_link_doublet_cluster(
                    df,
                    background_color=background_color,
                    title=title,
                    color_title=color_title
                )
                
                fig = plt.gcf()
                
                viz_placeholder.pyplot(fig)
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                buf.seek(0)
                
                download_placeholder.download_button(
                    "Download PNG",
                    data=buf,
                    file_name="clustering_visualization.png",
                    mime="image/png"
                )
                
                plt.close(fig)
                
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")

# link clustering page
elif st.session_state.current_page == "Link clustering":
    st.markdown("<h1 class='page-title'>Link Clustering</h1>", unsafe_allow_html=True)

    if st.session_state.dataframe_buffer is None:
        st.warning("Please upload data first.")
    else:
        df = pd.read_csv(io.StringIO(st.session_state.dataframe_buffer))

        if st.button("Cluster Links"):
            try:
                clusters = cluster_links(df)
                
                # Convert clusters to a DataFrame for display
                clusters_df = pd.DataFrame.from_dict(clusters, orient='index', columns=['Cluster ID'])
                clusters_df.index.name = 'Link'
                
                st.success("Links clustered successfully!")
                st.write("Cluster assignments:")
                st.dataframe(clusters_df)
                
                # Add download button
                clusters_csv = clusters_df.reset_index().to_csv(index=False)
                st.download_button(
                    "Download Cluster Assignments",
                    data=clusters_csv,
                    file_name="cluster_assignments.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error clustering links: {e}")