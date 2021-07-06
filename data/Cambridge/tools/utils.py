import numpy as np
import os, sys

def read_image_list(filename):
    print("Reading ImageList......")
    with open(filename, "r") as fin:
        lines = fin.readlines()
    lines = [line.strip() for line in lines]
    return lines