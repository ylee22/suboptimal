import pandas as pd

### Here just assume graph_dict is present since this is copy and pasted from a jupyter cell


def build_training_df(graph_dict):
    # Take a graph dictionary of edges and returns a pandas dataframe of training
    # data which has columns "source_id", "dest_id", "edge_count", 'source_subreddit', and 'dest_subreddit'
    # along with a dictionary
    # of subreddit names by id numbers, and a reverse dictionary of id numbers by subreddit names
    subreddit_by_id = {i: subreddit for i, subreddit in enumerate(sorted(graph_dict.keys()))}
    id_by_subreddit = {v: k for k, v in subreddit_by_id.items()}

    # Turn each link into a tuple which is added to a large dictionary, before turning
    # the final product into a data frame.
    rows = []
    for s1, d in graph_dict.items():
        s1_id = id_by_subreddit[s1]
        r = [(s1_id, id_by_subreddit[s2], d[s2]) for s2 in d.keys()]
        rows += r
    df = pd.DataFrame.from_records(rows, columns=["source_id", "dest_id", "edge_count"])
    # Add a column for the fraction of the edges coming into s1 in a given row. Can be done either
    # by modifying the construction to take the sum of d.values() in for loop, convert float and divide
    # or by using pandas group by source_id, compute total edge_count, join, then divide columns
    return df, id_by_subreddit, subreddit_by_id


def add_debugging_cols(graph_df, subreddit_by_id):
    # Takes a data frame with columns source_id and dest_id, and a dictionary of id to subreddit names.
    # Returns a data frame with additional columns: source_subreddit, and dest_subreddit based on looking up in the
    # dict.
    df = graph_df
    df["source_subreddit"] = df["source_id"].apply(lambda n: subreddit_by_id[n])
    df["dest_subreddit"] = df["dest_id"].apply(lambda n: subreddit_by_id[n])
    return df


# get rid of edges below a threshold
def prune_abs_edge(graph_df, min_edges=1):
    return graph_df[graph_df.edge_count > min_edges]


# calculate the edge frequency for each source subreddit
def calculate_frequency(graph_df):
    # group by source id
    total_subreddit_edges = graph_df.groupby('source_id').edge_count.sum().rename('total_edges')
    graph_df = graph_df.join(total_subreddit_edges, on='source_id')
    graph_df = graph_df.assign(edge_weight=graph_df['edge_count'] / graph_df['total_edges'])
    return graph_df


# calculate the conditional probability
def cond_prob_edge_freq(graph_df):
    # p(s1|s2) = p(s1 and s2)/p(s2)
    # total edge count = sum of all edge counts
    total_edge_count = sum(graph_df.edge_count)
    # p(s2) = total # s2 edges / total # of edges
    p_src = graph_df.groupby('source_id').edge_count.sum().rename('p_src') / total_edge_count
    graph_df = graph_df.join(p_src, on='source_id')
    # p(s1) = total # s1 edges / total # of edges
    p_dest = graph_df.groupby('dest_id').edge_count.sum().rename('p_dest') / total_edge_count
    graph_df = graph_df.join(p_dest, on='dest_id')

    # p(s1 and s2) = total # of s1 and s2 (edge_count) / total # of edges
    graph_df['p_dest_src'] = graph_df.edge_count / total_edge_count
    # conditional prob
    graph_df['p_dest_cond_src'] = graph_df.p_dest_src / graph_df.p_src
    graph_df['p_src_cond_dest'] = graph_df.p_dest_src / graph_df.p_dest
    return graph_df


# maximum number of connections to other subreddits
def prune_on_cond(graph_df, min_prob):
    # prune both ways p(dest|source) and p(source|dest)
    return graph_df[(graph_df.p_dest_cond_src > min_prob) & (graph_df.p_src_cond_dest > min_prob)]


def prune_top_n(graph_df, max_other):
    # return top n based on either conditional prob or raw edge count
    on_edge = graph_df.sort_values(by=['source_id', 'edge_count'], ascending=False).groupby('source_id').head(
        max_other).sort_values(by=['source_id', 'edge_count'])
    # conditional probability based on source id
    on_cond_prob = graph_df.sort_values(by=['source_id', 'p_dest_cond_src'], ascending=False).groupby('source_id').head(
        max_other).sort_values(by=['source_id', 'p_dest_cond_src'])
    # conditional probability based on dest id
    on_cond_prob = on_cond_prob.sort_values(by=['dest_id', 'p_src_cond_dest'], ascending=False).groupby('dest_id').head(
        max_other).sort_values(by=['dest_id', 'p_src_cond_dest'])
    return on_edge[['source_id', 'dest_id', 'edge_count']], on_cond_prob[['source_id', 'dest_id', 'p_dest_cond_src',
                                                                          'p_src_cond_dest']]


# remove personal user subreddits from keys
def remove_personal_subreddits(graph_dict):
    for k, d in list(graph_dict.items()):
        if k.startswith('u_'):
            graph_dict.pop(k)
        if isinstance(d, dict):
            remove_personal_subreddits(d)