#%%
import fire
import numpy as np
import regex as re
import os
from tqdm import tqdm
#%%

def lower_repl(match):
    s, w = match.groups()
    return f"{s}{'<up> ' if len(w) <= 2 else ''}{w.lower()}"

def handle_caps(x):
    return re.sub(r"(\b)([\p{Lu}\p{Lt}]+)", lower_repl, x).lower()

#print(handle_caps("Testujemy SUPER ciekawe zmiany dla PIotra"))
# -> <up>testujemy super ciekawe zmiany dla <up>piotra
def escape_caps(sentence_file, output):
    pbar = tqdm(total= os.path.getsize(sentence_file))
    with open(sentence_file, "r") as in_f:
        with open(output+".wrk", "w") as out_f:
            for line in in_f:
                out_f.write(handle_caps(line))
                pbar.update(len(line))
    pbar.close()
    os.rename(output+".wrk", output)
#%%
if __name__ == '__main__': fire.Fire(escape_caps)