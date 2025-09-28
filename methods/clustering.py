import networkx as nx
import networkx.algorithms.community as nx_comm

def cluster_links(df):
    """
    link clustering by Louvain's algorithm.

    parameters:
        df: DataFrame with 'from' and 'to' columns.

    returns:
        A dictionary of the form {'from->to': cluster_id}.
    """
    df['link'] = df['from'].astype(str) + '->' + df['to'].astype(str)

    G = nx.Graph()
    G.add_nodes_from(df['link'])

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            a_from, a_to = df.iloc[i]['from'], df.iloc[i]['to']
            b_from, b_to = df.iloc[j]['from'], df.iloc[j]['to']

            if ({a_from, a_to} & {b_from, b_to}):
                link_a = df.iloc[i]['link']
                link_b = df.iloc[j]['link']
                G.add_edge(link_a, link_b)

    partition = nx_comm.louvain_communities(G, resolution=1, seed=42)

    result = {}
    for cluster_id, nodes in enumerate(partition):
        for node in nodes:
            result[node] = cluster_id
    return result