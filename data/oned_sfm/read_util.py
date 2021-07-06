import numpy as np
import os

def read_list(list_file_path):

    with open(list_file_path) as f:
        lines = f.readlines()

    frame_list = []
    for i, line in enumerate(lines):
        if line.startswith('#'):
            continue

        tokens = line.split(' ')
        frame_list.append(tokens[0].strip())
    return frame_list