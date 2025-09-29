"""
deepvisual
classes:
- visualize_triplet_graph:
- visualize_doblet_graph:
- visualize_link_doublet: draws doublet links (the table format is from:to)
- visualize_link_doublet_cluster: clusters links using the Louvain method
"""

from methods.visualization import visualize_link_doublet, visualize_link_doublet_cluster

__version__ = "0.1.0"
__all__ = ['visualize_link_doublet', 'visualize_link_doublet_cluster']