import sentencepiece as spm
import numpy as np
import fire
#from concurrent.futures import ProcessPoolExecutor as Pool
from concurrent.futures import ThreadPoolExecutor as Pool
from fastai.text import partition_by_cores, num_cpus
import random
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from sys import stderr
from functools import reduce
from pickle import load
import regex as re

dec = re.compile("[\p{Lu}][^\p{Lu}]*")

def decapitalize(s):
  if dec.fullmatch(s):
    return s[0].lower() + s[1:]
  return s

def count_tokens(sentences):
  return Counter(token for sentence in sentences for token in sentence.split())
  
def escape_unknowns(sentences, vocabulary, unk_mark):
  def escape(sentence):
    return ' '.join([token if token in vocabulary else unk_mark for token in sentence.split()])
  return [escape(sentence)+'\n' for sentence in sentences]


def cap_to_dict(sentence_file, dictionary_file, output, lower_case=False, most_low=False, min_freq=3, unk_marker="☠"):
  with open(sentence_file, 'r') as f:
    lines = f.readlines()
  print(f"Cap-to-dict on file with {len(lines)} lines", file=stderr)
  with open(dictionary_file, 'rb') as f:
    tokens = load(f)

  if lower_case != most_low:
    print("lower_case != most_low is not supported", file=stderr)
    exit(1)

  vocabulary = set((decapitalize(token) if most_low else token) for token, freq in tokens.most_common() if freq >= min_freq)
  if lower_case:
    vocabulary.add('<up>')
  print(f"Dictionary was shrinked to {len(vocabulary)} tokens", file=stderr)
  lines = escape_unknowns(lines, vocabulary, unk_marker)
  output = Path(output)
  with open(output, 'w') as f:
    f.writelines(lines)

if __name__ == '__main__': fire.Fire(cap_to_dict)
