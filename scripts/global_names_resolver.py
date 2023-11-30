import requests
import json
from tqdm import tqdm
import argparse
'''
Example usage:

python global_names_resolver.py ../data/resolve-please.jsonl ../data/resolved.jsonl --batch_size 300

Note that performance seems to saturate beyond a batch size of 200-300 names.

This reads an input JSONL file with the following line format:
{
  "genus": "Trametes",
  "species": "trogii"
}

And outputs a JSONL file with the following line format:

{
  "Trametes trogii": {
    "Open Tree of Life Reference Taxonomy": {
      "kingdom": "Fungi",
      "phylum": "Basidiomycota",
      "class": "Agaricomycetes",
      "order": "Polyporales",
      "family": "Polyporaceae",
      "genus": "Trametes",
      "species": "Trametes trogii"
    }
  }
}

The output contains the search term used, the name of the provider for the highest scoring result, and the 7-rank hierarchy.
'''

# Function to resolve a batch of names and return structured data
def resolve_names(names):
    endpoint = 'http://resolver.globalnames.org/name_resolvers.json'
    params = {
        'names': '|'.join(names),
        'with_canonical_ranks': 'true'
    }
    response = requests.get(endpoint, params=params)
    if response.status_code != 200:
        return None

    data = response.json()
    structured_data_batch = {}

    for item_data in data['data']:  
        if 'results' not in item_data:  
            continue  

        highest_score = 0
        highest_score_data = {}
        for result in item_data['results']:
            score = result.get('score', 0)
            ranks = result.get('classification_path_ranks', '')
            desired_ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
            if all(rank in ranks.lower() for rank in desired_ranks):
                classification_path = result.get('classification_path', '')
                data_source_title = result.get('data_source_title', '')
                path_list = classification_path.split('|')
                rank_list = ranks.lower().split('|')
                if len(path_list) != len(rank_list):
                    continue
                rank_dict = {rank: term for rank, term in zip(rank_list, path_list) if rank in desired_ranks}
                if score > highest_score:
                    highest_score = score
                    highest_score_data = {data_source_title: rank_dict}
        name = item_data['supplied_name_string']
        if highest_score > 0:
            structured_data_batch[name] = highest_score_data

    return structured_data_batch

def main(input_file_path, output_file_path, batch_size=300):
    # Get total number of lines in the input file for tqdm
    with open(input_file_path, 'r') as infile:
        total_lines = sum(1 for line in infile)

    # Read input file and loop through each line
    with open(input_file_path, 'r') as infile:
        with open(output_file_path, 'w') as outfile:
            names_batch = []
            for line in tqdm(infile, total=total_lines, unit="line"):
                entry = json.loads(line)
                name = f"{entry['genus']} {entry['species']}"
                names_batch.append(name)
                if len(names_batch) == batch_size:
                    structured_data_batch = resolve_names(names_batch)
                    if structured_data_batch:
                        for name, data in structured_data_batch.items():
                            outfile.write(json.dumps({name: data}) + '\n')
                            outfile.flush()
                    names_batch = []  # Reset batch

            # Last batch if it has fewer names than the specified batch size
            if names_batch:
                structured_data_batch = resolve_names(names_batch)
                if structured_data_batch:
                    for name, data in structured_data_batch.items():
                        outfile.write(json.dumps({name: data}) + '\n')
                        outfile.flush()

    print("Data resolution complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resolve species names to structured data')
    parser.add_argument('input_file', help='Path to the input JSONL file')
    parser.add_argument('output_file', help='Path to the output JSONL file')
    parser.add_argument('--batch_size', type=int, default=300, help='Number of names to process in each batch')

    args = parser.parse_args()
    
    main(args.input_file, args.output_file, args.batch_size)

