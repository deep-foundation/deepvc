import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import networkx as nx
import networkx.algorithms.community as nx_comm

def visualize_link_doublet(
    df,
    loop_color='red',
    edge_color='black',
    inter_edge_color='blue',
    background_color='white',
    title='',
    color_title='black',
    figsize=(12, 8),
    show_labels=True
):

    existing_ids = set(df.index + 1)

    loops = df[df['from'] == df['to']].copy()
    first_level = df[(df['from'] != df['to']) & df['from'].isin(loops.index + 1) & df['to'].isin(loops.index + 1)].copy()
    second_level = df[~df.index.isin(loops.index) & ~df.index.isin(first_level.index)].copy()
    second_level = second_level[
        (second_level['from'].isin(first_level.index + 1) | second_level['from'].isin(loops.index + 1)) &
        (second_level['to'].isin(first_level.index + 1) | second_level['to'].isin(loops.index + 1))
    ]

    G = nx.DiGraph()
    nodes = pd.concat([loops['from'], loops['to'], first_level['from'], first_level['to']]).unique()
    G.add_nodes_from(nodes)
    G.add_edges_from(zip(first_level['from'], first_level['to']))
    pos = nx.circular_layout(G)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    operation_positions = {}

    def draw_infinity_loop(ax, x, y, label_id=None, size=0.12, color='red'):
        t = np.linspace(0, 2 * np.pi, 500)
        a = size
        x_loop = a * np.sin(t) / (1 + np.cos(t)**2)
        y_loop = a * np.sin(t) * np.cos(t) / (1 + np.cos(t)**2)
        x_loop += x
        y_loop += y
        ax.plot(x_loop, y_loop, color=color, lw=2)
        arrow_idx = -20
        arrow = FancyArrowPatch((x_loop[arrow_idx], y_loop[arrow_idx]),
                                (x_loop[arrow_idx + 1], y_loop[arrow_idx + 1]),
                                arrowstyle='->', color=color, mutation_scale=15, lw=2)
        ax.add_patch(arrow)
        if label_id is not None and show_labels:
            ax.text(x, y - 0.10, str(label_id), fontsize=10, color=color,
                    ha='center', va='center', zorder=5)

    for i, row in df.iterrows():
        src, dst = row['from'], row['to']
        op_id = i + 1

        if src == dst and src in pos:
            x, y = pos[src]
            draw_infinity_loop(ax, x, y, label_id=op_id, color=loop_color)

        elif src in pos and dst in pos:
            x1, y1 = pos[src]
            x2, y2 = pos[dst]
            arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                                    color=edge_color, mutation_scale=20, lw=2)
            ax.add_patch(arrow)

            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            if length != 0:
                ux, uy = dx / length, dy / length
                perp_x, perp_y = -uy, ux
                cross_length = 0.03
                start_x = x1 + ux * 0.07
                start_y = y1 + uy * 0.07
                line = Line2D(
                    [start_x - perp_x * cross_length, start_x + perp_x * cross_length],
                    [start_y - perp_y * cross_length, start_y + perp_y * cross_length],
                    color=edge_color, lw=2, zorder=1
                )
                ax.add_line(line)

            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            if show_labels:
                ax.text(mid_x + 0.03, mid_y + 0.03, str(op_id), fontsize=10,
                        color=edge_color, ha='left', va='bottom', zorder=5)
            operation_positions[op_id] = ((x1, y1), (x2, y2))

        elif ((src in operation_positions or src in pos) and
              (dst in operation_positions or dst in pos)):
            if src in operation_positions:
                (x1_start, y1_start), (x1_end, y1_end) = operation_positions[src]
                p1_x = (x1_start + x1_end) / 2
                p1_y = (y1_start + y1_end) / 2
            else:
                p1_x, p1_y = pos[src]

            if dst in operation_positions:
                (x2_start, y2_start), (x2_end, y2_end) = operation_positions[dst]
                p2_x = (x2_start + x2_end) / 2
                p2_y = (y2_start + y2_end) / 2
            else:
                p2_x, p2_y = pos[dst]

            rad = -0.9 if p1_y > p2_y else 0.9
            arrow = FancyArrowPatch((p1_x, p1_y), (p2_x, p2_y),
                                    connectionstyle=f"arc3,rad={rad}",
                                    arrowstyle='->', color=inter_edge_color,
                                    mutation_scale=20, lw=2)
            ax.add_patch(arrow)

            mid_x = (p1_x + p2_x) / 2
            mid_y = (p1_y + p2_y) / 2
            dx, dy = p2_x - p1_x, p2_y - p1_y
            length = np.hypot(dx, dy)
            perp_dx, perp_dy = -dy / length, dx / length
            offset = abs(rad) * 0.3
            arc_mid_x = mid_x + perp_dx * offset
            arc_mid_y = mid_y + perp_dy * offset
            if show_labels:
                ax.text(arc_mid_x, arc_mid_y, str(op_id), fontsize=10,
                        color=inter_edge_color, ha='center', va='center', zorder=5)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(title, color=color_title)
    plt.tight_layout()
    plt.show()

def visualize_link_doublet_cluster(df, background_color='white', title='', color_title='white'):
    """visualization of links with louvain clustering"""

    def cluster_links(df):
        """internal function for clustering links"""
        df['link'] = df['from'].astype(str) + '→' + df['to'].astype(str)
        G = nx.Graph()
        G.add_nodes_from(df['link'])

        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                a_from, a_to = df.iloc[i]['from'], df.iloc[i]['to']
                b_from, b_to = df.iloc[j]['from'], df.iloc[j]['to']
                if {a_from, a_to} & {b_from, b_to}:
                    G.add_edge(df.iloc[i]['link'], df.iloc[j]['link'])

        partition = nx_comm.louvain_communities(G, resolution=1, seed=42)
        clusters = {}
        for i, community in enumerate(partition):
            for node in community:
                clusters[node] = i
        return clusters

    clusters = cluster_links(df)

    G = nx.DiGraph()
    G.add_nodes_from(pd.concat([df['from'], df['to']]).unique())
    G.add_edges_from(zip(df['from'], df['to']))
    pos = nx.circular_layout(G)

    unique_clusters = set(clusters.values())
    palette = plt.cm.tab10.colors
    cluster_colors = {c: palette[i % len(palette)] for i, c in enumerate(unique_clusters)}

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    operation_positions = {}
    loops = df[df['from'] == df['to']]
    loop_nodes = set(loops['from'])

    for i, row in df.iterrows():
        src, dst = row['from'], row['to']
        link = f"{src}→{dst}"
        op_id = i + 1
        color = cluster_colors.get(clusters.get(link, -1), 'gray')

        if src == dst:
            t = np.linspace(0, 2*np.pi, 500)
            a = 0.12
            x_loop = a*np.sin(t)/(1+np.cos(t)**2) + pos[src][0]
            y_loop = a*np.sin(t)*np.cos(t)/(1+np.cos(t)**2) + pos[src][1]
            ax.plot(x_loop, y_loop, color=color, lw=2)

            arrow_idx = -20
            ax.add_patch(FancyArrowPatch(
                (x_loop[arrow_idx], y_loop[arrow_idx]),
                (x_loop[arrow_idx+1], y_loop[arrow_idx+1]),
                arrowstyle='->', color=color, mutation_scale=15, lw=2
            ))
            ax.text(pos[src][0], pos[src][1]-0.10, str(op_id),
                   fontsize=10, color=color, ha='center', va='center')

        elif src in loop_nodes and dst in loop_nodes:
            x1, y1 = pos[src]
            x2, y2 = pos[dst]

            ax.add_patch(FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='->', color=color,
                mutation_scale=20, lw=2
            ))

            dx, dy = x2-x1, y2-y1
            length = np.hypot(dx, dy)
            if length > 0:
                ux, uy = dx/length, dy/length
                perp_x, perp_y = -uy, ux
                start_x = x1 + ux*0.07
                start_y = y1 + uy*0.07
                ax.add_line(Line2D(
                    [start_x-perp_x*0.03, start_x+perp_x*0.03],
                    [start_y-perp_y*0.03, start_y+perp_y*0.03],
                    color=color, lw=2
                ))

            ax.text((x1+x2)/2+0.03, (y1+y2)/2+0.03, str(op_id),
                   fontsize=10, color=color, ha='left', va='bottom')
            operation_positions[op_id] = ((x1, y1), (x2, y2))

        else:
            if src in operation_positions and dst in operation_positions:
                (x1s, y1s), (x1e, y1e) = operation_positions[src]
                (x2s, y2s), (x2e, y2e) = operation_positions[dst]
                mid1 = ((x1s+x1e)/2, (y1s+y1e)/2)
                mid2 = ((x2s+x2e)/2, (y2s+y2e)/2)

                rad = -0.9 if mid1[1] > mid2[1] else 0.9
                ax.add_patch(FancyArrowPatch(
                    mid1, mid2,
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle='->', color=color,
                    mutation_scale=20, lw=2
                ))

                dx = mid2[0]-mid1[0]
                dy = mid2[1]-mid1[1]
                length = np.hypot(dx, dy)
                perp_dx, perp_dy = -dy/length, dx/length
                offset = 0.3*abs(rad)
                text_x = (mid1[0]+mid2[0])/2 + perp_dx*offset
                text_y = (mid1[1]+mid2[1])/2 + perp_dy*offset
                ax.text(text_x, text_y, str(op_id),
                       fontsize=10, color=color, ha='center', va='center')

    legend_elements = [Line2D([0], [0], color=c, lw=2, label=f'Cluster {k}')
                      for k, c in cluster_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(title, color=color_title)
    plt.tight_layout()
    plt.show()