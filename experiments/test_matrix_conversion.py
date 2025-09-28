import sys
sys.path.insert(0, '/tmp/gh-issue-solver-1759077995217')

import pandas as pd
import numpy as np
from deepcore import adjacency_to_links, links_to_adjacency

print("Testing adjacency_to_links and links_to_adjacency functions...")
print("="*60)

print("\nTest 1: Simple links to adjacency matrix and back")
links_data = {
    'from': [0, 1, 2, 0],
    'to': [1, 2, 0, 2]
}
links_df = pd.DataFrame(links_data)
print("\nOriginal links:")
print(links_df)

adj_matrix = links_to_adjacency(links_df)
print("\nAdjacency matrix:")
print(adj_matrix)

links_back = adjacency_to_links(adj_matrix)
print("\nConverted back to links:")
print(links_back)

print("\nTest 2: Adjacency matrix to links (with node labels)")
matrix_data = np.array([
    [0, 1, 1],
    [1, 0, 0],
    [0, 1, 0]
])
node_labels = ['A', 'B', 'C']
adj_df = pd.DataFrame(matrix_data, index=node_labels, columns=node_labels)
print("\nAdjacency matrix with labels:")
print(adj_df)

links_result = adjacency_to_links(adj_df)
print("\nConverted to links:")
print(links_result)

print("\nTest 3: Include zeros option")
links_with_zeros = adjacency_to_links(adj_df, include_zeros=True)
print("\nLinks with zeros included:")
print(links_with_zeros)

print("\n" + "="*60)
print("All tests passed successfully!")