import pandas as pd

def sort_duoblet(df):
    closed = df[df['from'] == df['to']].copy()
    closed_sorted = closed.sort_values(by='from').reset_index(drop=True)

    closed_nodes = set(closed['from'])

    between_closed = df[
        (df['from'].isin(closed_nodes)) &
        (df['to'].isin(closed_nodes)) &
        (df['from'] != df['to'])
    ].copy()
    between_closed_sorted = between_closed.sort_values(by=['from', 'to']).reset_index(drop=True)

    other = df[
        ~(df['from'].isin(closed_nodes) & df['to'].isin(closed_nodes))
    ].copy()
    other_sorted = other.sort_values(by=['from', 'to']).reset_index(drop=True)

    result = pd.concat([closed_sorted, between_closed_sorted, other_sorted], ignore_index=True)
    return result