import argparse
import cbor2
import json
import time

from pathlib import Path


def read_cbor(file: Path):
    with open(file, 'rb') as f:
        data = cbor2.load(f)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=Path, help='Input file')
    args = parser.parse_args()
    json_file = args.file.with_suffix('.json')

    start = time.time()
    data = read_cbor(args.file)
    print("CBOR: ", time.time() - start)

    # Dump the data to a JSON file
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

    # Read the JSON file
    start = time.time()
    with open(json_file, 'r') as f:
        data_json = json.load(f)
    print("JSON: ", time.time() - start)

    # Check if the data is the same
    assert data == data_json


if __name__ == '__main__':
    main()
