import os
import json
import gzip

DATA_FOLDER = "/home/shreyansh/embedding-training-data/"
DATA_CONFIG = "data_config.json"

with open(DATA_CONFIG) as fIn:
    data_config = json.load(fIn)

filepaths = []
dataset_indices = []
dc = {}

for idx, data in enumerate(data_config):
    filepaths.append((os.path.join(os.path.expanduser(DATA_FOLDER), data['name']), data['name']))

for filepath, orig_name in filepaths:
    print(orig_name)
    if "reddit_" in filepath:       #Special dataset class for Reddit files
        try:
            with gzip.open(filepath, "rt") as fIn:
                for line in fIn:
                    data = json.loads(line)
                    if "response" in data and "context" in data:
                        dc[orig_name] = 2
                        break
        except FileNotFoundError:
            print(f"****Not found {orig_name}")
    else:
        try:
            with gzip.open(filepath, "rt") as fIn:
                for line in fIn:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        try:
                            data = data['texts']
                        except KeyError:
                            print(data.keys())
                            break
                    dc[orig_name] = len(data)
                    break
        except FileNotFoundError:
            print(f"****Not found {orig_name}")
                
with open('num_cols.json', 'w') as fd:
    json.dump(dc, fd, indent=4)