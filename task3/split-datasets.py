import numpy as np
import fire
import random
from pathlib import Path


def split_to_numpy(ids_file, output_path, valid_size=10000000, test_set=False):
  with open(ids_file, 'r') as f:
    ids = [np.array([int(x) for x in line.split()]) for line in f.readlines()]

  output_path = Path(output_path)
  output_path.mkdir(parents=True, exist_ok=True)

  if test_set:
    np.save(output_path / 'test_ids.npy', np.array(ids))
    return

  random.seed(12345)
  random.shuffle(ids)
  valid = 0
  valid_idx = 0
  while valid < valid_size:
    valid += len(ids[valid_idx])
    valid_idx += 1
  
  train = 0
  train_idx = valid_idx
  while train < 123000000:
    train += len(ids[train_idx])
    train_idx += 1
#  train_idx = len(ids)

  np.save(output_path / 'val_ids.npy', np.array(ids[0:valid_idx]))
  np.save(output_path / 'trn_ids.npy', np.array(ids[valid_idx:train_idx]))

if __name__ == '__main__': fire.Fire(split_to_numpy)
