import streamlit as st
import pandas as pd
from pathlib import Path
from deepcore import sort_duoblet, adjacency_to_links, links_to_adjacency
from deepvisual import visualize_link_doublet
from deepcore import cluster_links
import matplotlib.pyplot as plt
import io
from deepvisual import visualize_link_doublet_cluster
import re
import numpy as np  

# --- set page config ---
st.set_page_config(
    page_title="DeepVC",
    layout="wide",
    page_icon="img/logo/logo.png"
)

# --- inject custom CSS ---
with open("style.css") as f:
    base_css = f.read()

# Add dark mode CSS if enabled
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if st.session_state.dark_mode:
    dark_mode_css = """
    body {
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
    }
    .page-title {
        color: #e0e0e0 !important;
    }
    .upload-info-box {
        background-color: #2c2c2c !important;
        color: #e0e0e0 !important;
        border: 1px solid #444444 !important;
    }
    .upload-info-box strong {
        color: #ffffff !important;
    }
    [data-testid="stDataFrame"] {
        background-color: #2c2c2c !important;
    }
    .home-content {
        background-color: #2c2c2c !important;
        color: #e0e0e0 !important;
    }
    """
    st.markdown(f"<style>{base_css}\n{dark_mode_css}</style>", unsafe_allow_html=True)
else:
    light_mode_css = """
    body {
        background-color: #ffffff !important;
        color: #2e3436 !important;
    }
    .page-title {
        color: #2e3436 !important;
    }
    .upload-info-box {
        background-color: #f0f0f0 !important;
        color: #2e3436 !important;
        border: 1px solid #cccccc !important;
    }
    .upload-info-box strong {
        color: #000000 !important;
    }
    .home-content {
        background-color: #f8f8f8 !important;
        color: #2e3436 !important;
    }
    """
    st.markdown(f"<style>{base_css}\n{light_mode_css}</style>", unsafe_allow_html=True)

# --- navigation control ---
# define pages in app
PAGES = {
    "Home": "Home",
    "Uploading data": "Uploading data",
    "Sorting the data": "Sorting the data",
    "Visualization of links": "Visualization of links",
    "Visualizing link clustering": "Visualizing link clustering",
    "Link clustering": "Link clustering",
    "Adjacency Matrix to Links": "Adjacency Matrix to Links",
    "Links to Adjacency Matrix": "Links to Adjacency Matrix"
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
if st.sidebar.button("Visualizing link clustering", key="vis_cluster_btn"):
    change_page("Visualizing link clustering")
if st.sidebar.button("Link clustering", key="cluster_btn"):
    change_page("Link clustering")

if st.sidebar.button("Adjacency Matrix to Links", key="adj_to_links_btn"):
    change_page("Adjacency Matrix to Links")
if st.sidebar.button("Links to Adjacency Matrix", key="links_to_adj_btn"):
    change_page("Links to Adjacency Matrix")

st.sidebar.markdown("""
<div class="nav-section">Triplet</div>
""", unsafe_allow_html=True)

# --- shared state via session_state ---
if "dataframe_buffer" not in st.session_state:
    st.session_state.dataframe_buffer = None
if "uploaded_file_info" not in st.session_state:
    st.session_state.uploaded_file_info = None
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def process_txt_content(content):
    """Обработка содержимого TXT файла и преобразование в DataFrame"""
    lines = content.split('\n')
    
    # (id: from to)
    pattern = re.compile(r'^\((\d+):\s*(\d+)\s+(\d+)\)$')

    # (final states)
    links = [
        (int(from_), int(to))
        for line in lines
        if (match := pattern.match(line.strip()))
        for _, from_, to in [match.groups()]
    ]

    # to dataframe
    return pd.DataFrame(links, columns=['from', 'to'])

def process_lino_content(content):
    lines = content.split('\n')
    pattern = re.compile(r"\(\d+:\s*(\d+)\s+(\d+)\)")
    data = [
        (int(src), int(tgt))
        for line in lines
        if (match := pattern.match(line.strip()))
        for src, tgt in [match.groups()]
    ]
    return pd.DataFrame(data, columns=["from", "to"])

# --- Theme Toggle Helper ---
def add_theme_toggle():
    col1, col2, col3 = st.columns([10, 1, 1])
    with col3:
        if st.button("🌙" if not st.session_state.dark_mode else "☀️", key=f"theme_toggle_{st.session_state.current_page}", help="Toggle dark/light mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

# --- page content ---
# home Page
if st.session_state.current_page == "Home":
    # add_theme_toggle()  # Removed as per issue #7
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
    # add_theme_toggle()  # Removed as per issue #7
    st.markdown("<h1 class='page-title'>Upload your data file</h1>", unsafe_allow_html=True)
    
    file_type = st.radio("Select file type:", ("CSV", "TXT", "LINO"), horizontal=True)
    
    if file_type == "CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader")
    elif file_type == "TXT":
        uploaded_file = st.file_uploader("Choose a TXT file", type=["txt"], key="txt_uploader")
    else:
        uploaded_file = st.file_uploader("Choose a LINO file", type=["lino"], key="lino_uploader")
    if uploaded_file is not None:
        try:
            if file_type == "CSV":
                df = pd.read_csv(uploaded_file)
            elif file_type == "TXT":
                content = uploaded_file.getvalue().decode("utf-8")
                df = process_txt_content(content)
            else:
                content = uploaded_file.getvalue().decode("utf-8")
                df = process_lino_content(content)
        
            st.session_state.dataframe_buffer = df.to_csv(index=False)
            st.session_state.uploaded_file_info = {
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "rows": len(df)
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
            if st.session_state.dataframe_buffer:
                df_preview = pd.read_csv(io.StringIO(st.session_state.dataframe_buffer))
                st.write("Preview (first 5 rows):")
                df_display = df_preview.head().copy()
                df_display.index = range(1, len(df_display) + 1)
                st.dataframe(df_display)
        except Exception as e:
            st.error(f"Error displaying data: {e}")

elif st.session_state.current_page == "Sorting the data":
    # add_theme_toggle()  # Removed as per issue #7
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
                df_display = sorted_df.head().copy()
                df_display.index = range(1, len(df_display) + 1)
                st.dataframe(df_display)
            except Exception as e:
                st.error(f"Error processing data: {e}")

        # download buttons for different formats
        st.download_button(
            "📥 Download Processed CSV",
            data=st.session_state.dataframe_buffer,
            file_name="processed.csv",
            mime="text/csv",
            key="download_csv_sorting"
        )

        st.download_button(
            "📥 Download Processed LINO",
            data="\n".join([f"({i}: {row['from']} {row['to']})" for i, row in df.iterrows()]),
            file_name="processed.lino",
            mime="text/plain",
            key="download_lino_sorting"
        )

        st.download_button(
            "📥 Download Processed TXT",
            data="\n".join([f"({i}: {row['from']} {row['to']})" for i, row in df.iterrows()]),
            file_name="processed.txt",
            mime="text/plain",
            key="download_txt_sorting"
        )

# visualization page
elif st.session_state.current_page == "Visualization of links":
    # add_theme_toggle()  # Removed as per issue #7
    st.markdown("<h1 class='page-title'>Visualization of Links</h1>", unsafe_allow_html=True)

    if st.session_state.dataframe_buffer is None:
        st.warning("Please upload and process data first.")
    else:
        df = pd.read_csv(io.StringIO(st.session_state.dataframe_buffer))

        col1, col2 = st.columns(2)
        with col1:
            loop_color = st.color_picker("Loop Color", "#FFA500")
            edge_color = st.color_picker("Edge Color", "#000000")
        with col2:
            inter_edge_color = st.color_picker("Inter Edge Color", "#0000FF")
            background_color = st.color_picker("Background Color", "#FFFFFF")

        title = st.text_input("Visualization Title", "")
        color_title = st.color_picker("Title Color", "#000000")
        
        st.markdown("**Figure Size (inches)**")
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            fig_width = st.number_input(
                "Width", 
                min_value=5.0, 
                max_value=20.0, 
                value=12.0, 
                step=0.5,
                help="Width of the figure in inches"
            )
        with fig_col2:
            fig_height = st.number_input(
                "Height", 
                min_value=5.0, 
                max_value=20.0, 
                value=8.0, 
                step=0.5,
                help="Height of the figure in inches"
            )
        figsize = (fig_width, fig_height)
        
        show_labels = st.checkbox("Show Labels", value=True)

        viz_placeholder = st.empty()
        download_placeholder = st.empty()

        if st.button("Generate Visualization"):
            try:
                plt.figure(figsize=figsize)
                
                visualize_link_doublet(
                    df,
                    loop_color=loop_color,
                    edge_color=edge_color,
                    inter_edge_color=inter_edge_color,
                    background_color=background_color,
                    title=title,
                    color_title=color_title,
                    show_labels=show_labels,
                    figsize=figsize
                )
                
                fig = plt.gcf()
                viz_placeholder.pyplot(fig)
                
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                buf.seek(0)
                
                download_placeholder.download_button(
                    "📥 Download PNG",
                    data=buf,
                    file_name="visualization.png",
                    mime="image/png"
                )
                
                plt.close(fig)
                
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")

# visualizing link clustering page
elif st.session_state.current_page == "Visualizing link clustering":
    # add_theme_toggle()  # Removed as per issue #7
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
                    "📥 Download PNG",
                    data=buf,
                    file_name="clustering_visualization.png",
                    mime="image/png"
                )
                
                plt.close(fig)
                
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")

# link clustering page
elif st.session_state.current_page == "Link clustering":
    # add_theme_toggle()  # Removed as per issue #7
    st.markdown("<h1 class='page-title'>Link Clustering</h1>", unsafe_allow_html=True)

    if st.session_state.dataframe_buffer is None:
        st.warning("Please upload data first.")
    else:
        df = pd.read_csv(io.StringIO(st.session_state.dataframe_buffer))

        if st.button("Cluster Links"):
            try:
                clusters = cluster_links(df)
                
                clusters_df = pd.DataFrame.from_dict(clusters, orient='index', columns=['Cluster ID'])
                clusters_df.index.name = 'Link'
                
                st.success("Links clustered successfully!")
                st.write("Cluster assignments:")
                st.dataframe(clusters_df)
                
                clusters_csv = clusters_df.reset_index().to_csv(index=False)
                st.download_button(
                    "📥 Download Cluster Assignments",
                    data=clusters_csv,
                    file_name="cluster_assignments.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error clustering links: {e}")

elif st.session_state.current_page == "Adjacency Matrix to Links":
    # add_theme_toggle()  # Removed as per issue #7
    st.markdown("<h1 class='page-title'>Adjacency Matrix to Links</h1>", unsafe_allow_html=True)

    st.markdown("Upload an adjacency matrix file (CSV format)")

    uploaded_file = st.file_uploader("Choose a CSV file with adjacency matrix", type=["csv"], key="adj_matrix_uploader")

    include_zeros = st.checkbox("Include zero-weight links", value=False)

    if uploaded_file is not None:
        try:
            adj_matrix_df = pd.read_csv(uploaded_file, index_col=0)

            st.write("Uploaded Adjacency Matrix:")
            adj_display = adj_matrix_df.copy()
            adj_display.index = range(1, len(adj_display) + 1)
            st.dataframe(adj_display)

            if st.button("Convert to Links"):
                try:
                    links_df = adjacency_to_links(adj_matrix_df, include_zeros=include_zeros)

                    st.success("Conversion successful!")
                    st.write("Result (Links):")
                    links_display = links_df.copy()
                    links_display.index = range(1, len(links_display) + 1)
                    st.dataframe(links_display)

                    links_csv = links_df.to_csv(index=False)

                    st.download_button(
                        "📥 Download CSV",
                        data=links_csv,
                        file_name="links.csv",
                        mime="text/csv",
                        key="download_csv_adj_to_links"
                    )

                    st.download_button(
                        "📥 Download LINO",
                        data="\n".join([f"({i}: {row['from']} {row['to']})" for i, row in links_df.iterrows()]),
                        file_name="links.lino",
                        mime="text/plain",
                        key="download_lino_adj_to_links"
                    )

                    st.download_button(
                        "📥 Download TXT",
                        data="\n".join([f"({i}: {row['from']} {row['to']})" for i, row in links_df.iterrows()]),
                        file_name="links.txt",
                        mime="text/plain",
                        key="download_txt_adj_to_links"
                    )

                except Exception as e:
                    st.error(f"Error converting matrix to links: {e}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

elif st.session_state.current_page == "Links to Adjacency Matrix":
    # add_theme_toggle()  # Removed as per issue #7
    st.markdown("<h1 class='page-title'>Links to Adjacency Matrix</h1>", unsafe_allow_html=True)

    if st.session_state.dataframe_buffer is None:
        st.warning("Please upload data first in 'Uploading data' page.")
    else:
        df = pd.read_csv(io.StringIO(st.session_state.dataframe_buffer))

        st.write("Uploaded Links:")
        df_display = df.head().copy()
        df_display.index = range(1, len(df_display) + 1)
        st.dataframe(df_display)

        weight_column = None
        if len(df.columns) > 2:
            use_weights = st.checkbox("Use weight column", value=False)
            if use_weights:
                weight_column = st.selectbox("Select weight column:", [col for col in df.columns if col not in ['from', 'to']])

        if st.button("Convert to Adjacency Matrix"):
            try:
                adj_matrix_df = links_to_adjacency(df, weight_column=weight_column)

                st.success("Conversion successful!")
                st.write("Result (Adjacency Matrix):")
                adj_display = adj_matrix_df.copy()
                adj_display.index = range(1, len(adj_display) + 1)
                st.dataframe(adj_display)

                adj_matrix_csv = adj_matrix_df.to_csv()

                st.download_button(
                    "📥 Download CSV",
                    data=adj_matrix_csv,
                    file_name="adjacency_matrix.csv",
                    mime="text/csv",
                    key="download_csv_links_to_adj"
                )

                links_from_matrix = adjacency_to_links(adj_matrix_df, include_zeros=False)
                st.download_button(
                    "📥 Download LINO",
                    data="\n".join([f"({i}: {row['from']} {row['to']})" for i, row in links_from_matrix.iterrows()]),
                    file_name="adjacency_matrix.lino",
                    mime="text/plain",
                    key="download_lino_links_to_adj"
                )

                st.download_button(
                    "📥 Download TXT",
                    data="\n".join([f"({i}: {row['from']} {row['to']})" for i, row in links_from_matrix.iterrows()]),
                    file_name="adjacency_matrix.txt",
                    mime="text/plain",
                    key="download_txt_links_to_adj"
                )

            except Exception as e:
                st.error(f"Error converting links to matrix: {e}")