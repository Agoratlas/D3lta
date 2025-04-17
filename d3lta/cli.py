import argparse
import sys
import os
import pandas as pd
import collections

from .faissd3lta import semantic_faiss

def export_summary(df_clusters, output_file, text_column_name, top_n_examples=5):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('D3lta analysis report\n')
        f.write('========================\n')
        f.write(f'Total number of clusters:           {df_clusters["cluster"].nunique()}\n')
        f.write(f'Total number of documents:          {len(df_clusters)}\n')
        f.write(f'Documents identified as duplicates: {len(df_clusters[~pd.isna(df_clusters["cluster"])])}\n')
        f.write(f'Largest clusters:\n')
        cluster_cnt = df_clusters['cluster'].value_counts()
        for cluster_id, count in cluster_cnt.head(top_n_examples).items():
            f.write(f'  - Cluster {cluster_id}: {count} documents\n')
        f.write('========================\n')
        for cluster_id, group in df_clusters.groupby('cluster', sort=True):
            f.write(f'Cluster {cluster_id}:\n')
            f.write(f'  Size: {len(group)} documents\n')

            most_present_docs = collections.Counter(group[text_column_name]).most_common(top_n_examples)
            f.write('  Examples:\n')
            for doc, count in most_present_docs:
                if count > 1:
                    f.write(f'    - "{doc}" ({count} times)\n')
                else:
                    f.write(f'    - "{doc}"\n')
            f.write('\n')


def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(
        description='Command-line utility for D3lta, a library for detecting duplicate verbatim contents within a vast amount of documents.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Add arguments
    parser.add_argument('input_file', type=str, help='A CSV file containing the documents to process.')

    parser.add_argument('-c', '--column', type=str, help='The column name containing the text to process.')
    parser.add_argument('-cn', '--column-number', type=int, help='For CSV files without a header, the (zero-indexed) column number containing the text to process.')

    parser.add_argument('-o', '--output-directory', type=str, help='Output directory for the results (default: current directory).', default='.')
    parser.add_argument(
        '-t', '--output-type', 
        default='full', 
        help=(
            'Specify the type of output to generate. Available options are:\n'
            '  tagged  - Outputs the provided CSV with an additional column containing the cluster number of each document.\n'
            '  matches - Outputs the list of all detected matches.\n'
            '  graph   - Outputs a simplified graph representation of documents.\n'
            '  summary - Outputs a summary of the detected clusters, with their size and a few examples.\n'
            '  full    - All of the above. (default)\n'
        )
    )

    parser.add_argument('--min-size-txt', type=int, default=30, help='Minimum size for each document to be processed (default: 30).')

    # Thresholds (grapheme, language, and semantic)
    parser.add_argument('--threshold-grapheme', type=float, default=0.693, help='Threshold for grapheme similarity in copypasta detection (default: 0.693).')
    parser.add_argument('--threshold-language', type=float, default=0.715, help='Threshold for language similarity in translation detection (default: 0.715).')
    parser.add_argument('--threshold-semantic', type=float, default=0.85, help='Threshold for semantic similarity in rewording detection (default: 0.85).')
    
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        parser.error(f'The input file {args.input_file} does not exist.')
    if not args.input_file.endswith('.csv'):
        parser.error('The input file must be a .csv file.')

    if args.column is not None and args.column_number is not None:
        parser.error('You cannot specify both --column and --column-number.')

    if args.column is None and args.column_number is None:
        parser.error('You must specify either --column or --column-number.')
        
    output_filenames = {}

    if args.output_type in ('tagged', 'full'):
        output_filenames['tagged'] = os.path.join(args.output_directory, 'd3lta_tagged_' + args.input_file)
    if args.output_type in ('matches', 'full'):
        output_filenames['matches'] = os.path.join(args.output_directory, 'd3lta_matches_' + args.input_file)
    if args.output_type in ('graph', 'full'):
        output_filenames['graph'] = os.path.join(args.output_directory, 'd3lta_graph_' + args.input_file)
    if args.output_type in ('summary', 'full'):
        output_filenames['summary'] = os.path.join(args.output_directory, 'd3lta_summary_' + args.input_file + '.txt')

    # Check if any output file already exists
    if any(os.path.isfile(filename) for filename in output_filenames.values()):
        response = input('Warning: Some output files already exist. Do you want to overwrite them? [y/N] ')
        if response.lower() != 'y':
            print('Aborting.')
            sys.exit(0)
    
    os.makedirs(args.output_directory, exist_ok=True)

    input_has_header = (args.column is not None)
    if input_has_header:
        input_df = pd.read_csv(args.input_file, encoding='utf-8', dtype=str)
        text_column_name = args.column
        if text_column_name not in input_df.columns:
            parser.error(f'The specified column "{text_column_name}" does not exist in the input file.')
    else:
        input_df = pd.read_csv(args.input_file, header=None, encoding='utf-8', dtype=str)
        input_df.columns = [f'col_{i}' for i in range(input_df.shape[1])]
        text_column_name = f'col_{args.column_number}'
        if text_column_name not in input_df.columns:
            parser.error(f'The input file has columns numbered from 0 to {len(input_df.columns)-1}, but the specified column number {args.column_number} is out of range.')
    
    input_df.index = input_df.index.astype(str)

    input_df[text_column_name] = input_df[text_column_name].fillna('')

    tagged_df = input_df.copy()

    matches, df_clusters = semantic_faiss(
        df=input_df,
        min_size_txt=args.min_size_txt,
        threshold_grapheme=args.threshold_grapheme,
        threshold_language=args.threshold_language,
        threshold_semantic=args.threshold_semantic,
        text_column=text_column_name,
    )

    if 'tagged' in output_filenames:
        tagged_df['cluster'] = df_clusters['cluster']
        tagged_df.to_csv(output_filenames['tagged'], index=False, encoding='utf-8', header=input_has_header)
        print(f'Tagged output saved to {output_filenames["tagged"]}')
    if 'matches' in output_filenames:
        matches.to_csv(output_filenames['matches'], index=False, encoding='utf-8')
        print(f'Matches output saved to {output_filenames["matches"]}')
    if 'graph' in output_filenames:
        df_clusters.to_csv(output_filenames['graph'], index=False, encoding='utf-8')
        print(f'Graph output saved to {output_filenames["graph"]}')
    if 'summary' in output_filenames:
        export_summary(df_clusters, output_filenames['summary'], text_column_name)
        print(f'Summary output saved to {output_filenames["summary"]}')
    
    print('Processing completed successfully.')
    return 0


if __name__ == '__main__':
    sys.exit(main())