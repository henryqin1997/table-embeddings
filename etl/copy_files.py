import json
import sys
import shutil
import os

files_json = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]

if __name__ == '__main__':
    files = json.load(open(files_json))
    if type(files) is dict:
        files = files.values()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in files:
        shutil.copyfile(os.path.join(input_dir, file), os.path.join(output_dir, file))
