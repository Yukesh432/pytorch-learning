import io
import os
import unicodedata
import string
import torch
import random
import glob

ALL_LETTERS= string.ascii_letters+ ",.;"
NUM_LETTERS= len(ALL_LETTERS)

print(NUM_LETTERS)
print(ALL_LETTERS)

# convert unicode string to ASCII

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c)!='Mn' and c in ALL_LETTERS)

def load_data():
    category_lines= {}
    all_categories= []

    def find_files(path):
        return glob.glob(path)
    
    # read a file and split into lines
    def read_lines(filename):
        lines= io.open(filename, encoding='utf-8').read().strip().split('\n')
