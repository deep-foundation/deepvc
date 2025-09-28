import pandas as pd
import numpy as np

def adjacency_to_links(adjacency_matrix, node_labels=None, include_zeros=False):
    """
    converts an adjacency matrix to associative links format.

    parameters:
        adjacency_matrix: numpy array or DataFrame representing the adjacency matrix.
        node_labels: list of node labels. If None, uses indices.
        include_zeros: if True, includes links with zero weight. Default is False.

    returns:
        DataFrame with columns 'from', 'to', and optionally 'weight'.
    """
    if isinstance(adjacency_matrix, pd.DataFrame):
        if node_labels is None:
            node_labels = adjacency_matrix.columns.tolist()
        matrix = adjacency_matrix.values
    else:
        matrix = np.array(adjacency_matrix)
        if node_labels is None:
            node_labels = list(range(matrix.shape[0]))

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency matrix must be square")

    if len(node_labels) != matrix.shape[0]:
        raise ValueError("node_labels length must match matrix dimensions")

    links = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            weight = matrix[i, j]
            if include_zeros or weight != 0:
                links.append({
                    'from': node_labels[i],
                    'to': node_labels[j],
                })

    df = pd.DataFrame(links)

    df = df.iloc[df.apply(lambda x: (x['from'] != x['to'], x['from'], x['to']), axis=1).argsort()].reset_index(drop=True)

    return df

def links_to_adjacency(df, weight_column=None):
    """
    converts associative links to an adjacency matrix.

    parameters:
        df: DataFrame with 'from' and 'to' columns.
        weight_column: name of the column to use as edge weights. If None, uses 1 for all edges.

    returns:
        DataFrame representing the adjacency matrix with node labels as index and columns.
    """
    if 'from' not in df.columns or 'to' not in df.columns:
        raise ValueError("DataFrame must contain 'from' and 'to' columns")

    all_nodes = sorted(set(df['from'].unique()) | set(df['to'].unique()))
    n = len(all_nodes)

    node_to_idx = {node: i for i, node in enumerate(all_nodes)}

    matrix = np.zeros((n, n))

    for _, row in df.iterrows():
        i = node_to_idx[row['from']]
        j = node_to_idx[row['to']]
        weight = row[weight_column] if weight_column and weight_column in df.columns else 1
        matrix[i, j] = weight

    adjacency_df = pd.DataFrame(matrix, index=all_nodes, columns=all_nodes)
    return adjacency_df