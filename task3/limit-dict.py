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


def count_tokens(sentences):
  return Counter(token for sentence in sentences for token in sentence.split())
  
def escape_unknowns(sentences, vocabulary, unk_mark):
  def escape(sentence):
    return ' '.join([token if token in vocabulary else unk_mark for token in sentence.split()])
  return [escape(sentence)+'\n' for sentence in sentences]


def limit_dict(sentence_file, output_path, min_freq=3, threads=8, unk_marker="☠"):
  with open(sentence_file, 'r') as f:
    lines = f.readlines()
  print(f"Read {len(lines)} lines", file=stderr)
  tasks = partition_by_cores(lines)
  with Pool(threads) as e:
    counters = e.map(count_tokens, tasks)
    tokens = reduce(lambda a, b: a + b, counters)

  vocabulary = set(token for token, freq in tokens.most_common() if freq >= min_freq)
  path = Path(output_path)
  with open(path / 'vocabulary.txt', 'w') as f:
    f.writelines([v + '\n' for v in vocabulary])

  print(f"Created dictionary with {len(vocabulary)} tokens", file=stderr)
  with Pool(threads) as e:
    lines = sum(e.map(escape_unknowns, tasks, [vocabulary]*len(tasks), [unk_marker] * len(tasks)), [])
  with open(path / 'escaped.txt', 'w') as f:
    f.writelines(lines)

if __name__ == '__main__': fire.Fire(limit_dict)
