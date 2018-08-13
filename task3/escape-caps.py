#%%
import fire
import numpy as np
import regex as re
import os
from tqdm import tqdm
#%%
def handle_caps_just_low(line):
    def lower_repl(match):
        s, w = match.groups()
        return f"{s}{'<up> ' if len(w) <= 2 else ''}{w.lower()}"

    def handle_caps_in_token(x):
        return re.sub(r"(^| )([\p{Lu}\p{Lt}]+)", lower_repl, x).lower()

    return ' '.join([handle_caps_in_token(token) for token in line.split(' ')])

def handle_caps_most_low(line):
    def handle_caps_in_token(x):
        if re.fullmatch(r"^[\p{Lu}\p{Lt}][\p{Ll}]+\n?$", x):
            return f"<up> {x.lower()}"
        return x
    return ' '.join([handle_caps_in_token(token) for token in line.split(' ')])

#print(handle_caps("Testujemy SUPER ciekawe zmiany dla PIotra"))
# -> <up>testujemy super ciekawe zmiany dla <up>piotra
def escape_caps(sentence_file, output,  most_low=False):
    pbar = tqdm(total= os.path.getsize(sentence_file))
    with open(sentence_file, "r") as in_f:
        with open(output+".wrk", "w") as out_f:
            for line in in_f:
                if most_low:
                    out_f.write(handle_caps_most_low(line))
                else:
                    out_f.write(handle_caps_just_low(line))
                pbar.update(len(line))
    pbar.close()
    os.rename(output+".wrk", output)
#%%
if __name__ == '__main__': fire.Fire(escape_caps)