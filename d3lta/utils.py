
import collections
import csv
import pandas as pd


def get_cluster_statistics(df_clusters, text_column_name, top_n_examples=5):
    """
    Extract structured statistics about clusters from D3LTA results.
    
    This function separates data extraction from formatting, returning
    raw cluster information that can be used by export_summary() or
    for custom display/reporting.
    
    Args:
        df_clusters (pd.DataFrame): DataFrame with cluster assignments
        text_column_name (str): Name of the column containing the text
        top_n_examples (int): Number of top examples to extract per cluster (default: 5)
    
    Returns:
        dict: Dictionary with 'global_stats' and 'clusters' keys:
            - global_stats: dict with n_clusters, n_documents, n_tagged
            - clusters: list of dicts (sorted by size desc), each with:
                * cluster_id: the cluster identifier
                * size: number of documents in cluster
                * examples: list of dicts with 'text' and 'count'
    """
    # Global statistics
    n_clusters = df_clusters['cluster'].nunique()
    n_tagged = len(df_clusters[~pd.isna(df_clusters['cluster'])])
    
    global_stats = {
        'n_clusters': n_clusters,
        'n_documents': len(df_clusters),
        'n_tagged': n_tagged
    }
    
    # Cluster-specific information
    clusters = []
    cluster_counts = df_clusters['cluster'].value_counts()
    
    for cluster_id in cluster_counts.index:
        if pd.isna(cluster_id):
            continue
            
        group = df_clusters[df_clusters['cluster'] == cluster_id]
        size = len(group)
        
        # Get most common documents in this cluster
        most_common_docs = collections.Counter(
            group[text_column_name]
        ).most_common(top_n_examples)
        
        examples = [
            {'text': doc, 'count': count}
            for doc, count in most_common_docs
        ]
        
        clusters.append({
            'cluster_id': cluster_id,
            'size': size,
            'examples': examples
        })
    
    # Sort clusters by size (descending)
    clusters.sort(key=lambda x: x['size'], reverse=True)
    
    return {
        'global_stats': global_stats,
        'clusters': clusters
    }


def export_summary(df_clusters, output_file, text_column_name,
                    top_n_examples=5):
    """Export a summary of the clusters to a text file.

    This contains a simplified view with high-level statistics
    (e.g. number of clusters, number of documents, etc.)
    and a few examples of documents within each cluster.

    Args:
        df_clusters (pd.DataFrame): DataFrame containing the clusters.
        output_file (str): Path to the output file.
        text_column_name (str): Name of the column containing the text.
        top_n_examples (int): Number of examples to show for each cluster.
    """
    stats = get_cluster_statistics(df_clusters, text_column_name, top_n_examples)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write global statistics
        f.write('D3lta analysis report\n')
        f.write('========================\n')
        f.write(f'Total number of clusters:           {stats["global_stats"]["n_clusters"]}\n')
        f.write(f'Total number of documents:          {stats["global_stats"]["n_documents"]}\n')
        f.write(f'Documents identified as duplicates: {stats["global_stats"]["n_tagged"]}\n')
        f.write('Largest clusters:\n')
        
        for cluster in stats['clusters'][:top_n_examples]:
            f.write(f'  - Cluster {cluster["cluster_id"]}: {cluster["size"]} documents\n')
        
        f.write('========================\n')
        
        # Write detailed cluster information
        for cluster in stats['clusters']:
            f.write(f'Cluster {cluster["cluster_id"]}:\n')
            f.write(f'  Size: {cluster["size"]} documents\n')
            f.write('  Examples:\n')
            
            for example in cluster['examples']:
                if example['count'] > 1:
                    f.write(f'    - "{example["text"]}" ({example["count"]} times)\n')
                else:
                    f.write(f'    - "{example["text"]}"\n')
            f.write('\n')


def export_graph(df_clusters, matches, output_file, text_column_name, id_column_name):
    """Export a simplified graph representation of the clusters to a CSV file.

    Each cluster is represented as a node, and each document is connected
    to its cluster. The cluster's label is determined as the most "central"
    document, i.e. the one with highest total match score.

    Args:
        df_clusters (pd.DataFrame): DataFrame containing the clusters.
        matches (pd.DataFrame): DataFrame containing the matches.
        output_file (str): Path to the output file.
        text_column_name (str): Name of the column containing the text.
    """
    node_centrality = collections.defaultdict(float)
    for _, row in matches.iterrows():
        node_centrality[row['source']] += row['score']
        node_centrality[row['target']] += row['score']

    with open(output_file, 'w', encoding='utf-8') as f:
        csv_fields = ['source_id', 'source_label', 'target_id', 'target_label']
        graph_csv = csv.DictWriter(f, fieldnames=csv_fields)
        graph_csv.writeheader()
        for cluster_value, group in df_clusters.groupby('cluster', sort=True):
            cluster_id = f'cluster_{cluster_value}'
            # Find the node with highest node_centrality within the group
            central_node_id = max(
                group.index,
                key=lambda node_id: node_centrality[node_id]
            )

            cluster_label = group.loc[central_node_id, text_column_name]
            for _, row in group.iterrows():
                graph_csv.writerow({
                    'source_id': row[id_column_name],
                    'source_label': row[text_column_name],
                    'target_id': cluster_id,
                    'target_label': cluster_label
                })