import sentencepiece as spm
import numpy as np
import fire
import random
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from sys import stderr
from functools import reduce
from pickle import dump

def count_tokens(sentences):
  return Counter(token for sentence in sentences for token in sentence.split())
  

def extract_dict(sentence_file, output, min_freq=3, unk_marker="☠"):
  with open(sentence_file, 'r') as f:
    lines = f.readlines()
  print(f"Extracting dictionary from {len(lines)} lines", file=stderr)
  counter = count_tokens(lines)

  path = Path(output)
  with open(path, 'wb') as f:
    dump(counter, f)

if __name__ == '__main__': fire.Fire(extract_dict)
