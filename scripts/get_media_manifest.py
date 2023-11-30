import os
import re
import requests
import tarfile
import argparse
import hashlib
import json
from datetime import datetime
from tempfile import TemporaryDirectory
from tqdm import tqdm

'''
Usage:
$ python get_media_manifest.py --output_dir /absolute/path/to/output_dir
$ python get_media_manifest.py --output_dir ../relative/path/to/output_dir
$ python get_media_manifest.py # saves the files in the current directory

Input (optional): --output_dir /path/to/output_dir
Outputs: 
- eol_media_manifest_YYYY-MM-DD.csv
- eol_media_manifest_YYYY-MM-DD_log.json

This script downloads the EOL media manifest from https://eol.org/data/media_manifest.tgz, extracts the CSV files from the tarball, combines them into a single CSV file, and creates JSON log file.

The JSON log contains:
- The start time of the download
- The MD5 checksum of the temporary TGZ file
- The MD5 checksum of the saved combined CSV file
- The path to the saved combined CSV file
- The column names of the CSV file
- The total number of entries in the combined CSV file

The script will create the output directory if it does not exist upon confirmation by the user.
If the output directory is not specified, the script will save the combined CSV file and log file in the current directory.
'''

DOWNLOAD_URL = "https://eol.org/data/media_manifest.tgz"


def stream_download_file(file_path):
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with requests.get(DOWNLOAD_URL, stream=True) as response:
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading")

        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                progress_bar.update(len(chunk))
                file.write(chunk)
        progress_bar.close()
    
    md5_checksum = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
    return start_time, md5_checksum

# Extract the numeric part from the chunked manifest file names
def extract_numeric_part(file_name):
    match = re.search(r'\d+', file_name)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No numeric part found in {file_name}")

def generate_output_filename(output_dir):
    today = datetime.today().strftime('%Y-%m-%d')
    filename = f"eol_media_manifest_{today}.csv"
    # If the file exists in this location, append a number to the filename
    i = 1
    while os.path.exists(os.path.join(output_dir, filename)):
        filename = f"eol_media_manifest_{today}_{i}.csv"
        i += 1
    return os.path.join(output_dir, filename)

def extract_and_combine_csv(tgz_file_path, output_file_path):
    with tarfile.open(tgz_file_path, 'r:gz') as tar:
        sorted_members = sorted(tar.getmembers(), key=lambda member: extract_numeric_part(member.name))

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            headers_written = False
            total_lines = 0
            for member in tqdm(sorted_members, desc="Extracting files"):
                if member.name.startswith("media_manifest_") and member.name.endswith(".csv"):
                    f = tar.extractfile(member)
                    if f:
                        lines = f.read().decode('utf-8').splitlines()
                        if not headers_written:
                            headers = lines[0]
                            headers_written = True
                        output_file.write('\n'.join(lines))
                        output_file.write('\n')
                        total_lines += len(lines) - 1
    return headers, total_lines

def create_json_log(log_filename, download_start_time, tgz_md5, csv_md5, output_file_path, headers, total_lines):
    log_data = {
        "download_start_time": download_start_time,
        "temp_tgz_md5": tgz_md5,
        "manifest_csv_md5": csv_md5,
        "manifest_csv_filepath": os.path.abspath(output_file_path),
        "column_names": headers.split(','),
        "total_entries_in_media_manifest": total_lines
    }

    with open(log_filename, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)

def main(output_dir):
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        confirm = input(f"The directory {output_dir} does not exist. Do you want it to be created now? (y/n): ")
        if confirm.lower() != 'y':
            print("Directory creation cancelled. Exiting script.")
            return
        else:
            os.makedirs(output_dir)
    
    output_file_path = generate_output_filename(output_dir)
    base_output_file_name = os.path.splitext(output_file_path)[0]
    log_filename = f"{base_output_file_name}_log.json"


    with TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "media_manifest.tgz")
        download_start_time, tgz_md5 = stream_download_file(temp_file_path)

        headers, total_lines = extract_and_combine_csv(temp_file_path, output_file_path)

        csv_md5 = hashlib.md5(open(output_file_path, 'rb').read()).hexdigest()

    create_json_log(log_filename, download_start_time, tgz_md5, csv_md5, output_file_path, headers, total_lines)

    print(f"EOL media manifest: {output_file_path}")
    print(f"Log file:           {log_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download, combine, and log EOL media manifest CSV files.")
    parser.add_argument('--output_dir', type=str, default=".",
                        help="Directory to store the combined CSV file and log file. Defaults to the current directory.")
    args = parser.parse_args()

    main(args.output_dir)
    
