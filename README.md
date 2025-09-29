# DeepVC - Deep Visual Core

A powerful Python library and web application for working with associative link data structures, providing visualization, analysis, and conversion tools for links notation (<a href='https://github.com/linksplatform/Protocols.Lino'>LINO</a>) data.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deepvc.streamlit.app/)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](LICENSE)

## Features

- **Link Data Visualization**: Create beautiful visualizations of doublet link structures with customizable colors and layouts
- **Link Clustering**: Apply Louvain algorithm for community detection in link networks
- **Data Sorting**: Sort doublet data structures based on closed links and relationships
- **Matrix Conversion**: Convert between adjacency matrices and link representations
- **Interactive Web UI**: Streamlit-based interface for easy data exploration and analysis
- **Multiple Export Formats**: Export results to CSV, TXT, and LINO formats

## Quick Start

### Web Application

Visit the live demo: [https://deepvc.streamlit.app/](https://deepvc.streamlit.app/)

### Installation

```bash
pip install -r requirements.txt
```

### Usage

Run the Streamlit app locally:
```bash
streamlit run app.py
```

## Core Functions

### Data Processing
- `sort_duoblet(df)`: Sort link data by closed links and relationships
- `cluster_links(df)`: Cluster links using Louvain algorithm

### Matrix Conversion
- `adjacency_to_links(matrix, node_labels, include_zeros)`: Convert adjacency matrix to link format
- `links_to_adjacency(df, weight_column)`: Convert links to adjacency matrix

### Visualization
- `visualize_link_doublet(df, ...)`: Visualize doublet link structures
- `visualize_link_doublet_cluster(df, ...)`: Visualize with clustering

## License

MIT License - see [LICENSE](LICENSE) file for details
