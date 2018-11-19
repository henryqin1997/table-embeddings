import json
import os
import random

random.seed(10)

num_folders = 51
all_files = []

if __name__ == '__main__':
    for folder in range(num_folders):
        folder = str(folder).zfill(2)
        print(folder)
        for file in os.listdir(os.path.join('webtables', folder)):
            if random.random() < 0.001:
                all_files.append(os.path.join('webtables', folder, file))
    json.dump(all_files, open('temp_filelist.json', 'w+'), indent=4)
