import json
import sys
import shutil
import os

files_json = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
suffixes = ['.json', '_date.csv', '_ner.csv', '_nst.csv', '_wordlist.csv']

if __name__ == '__main__':
    files = json.load(open(files_json))
    if type(files) is dict:
        files = files.values()

    os.makedirs(output_dir, exist_ok=True)

    for file in files:
        dir_name, _ = os.path.split(file)
        os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
        for suffix in suffixes:
            shutil.copyfile(os.path.join(input_dir, os.path.splitext(file)[0] + suffix),
                            os.path.join(output_dir, os.path.splitext(file)[0] + suffix))
