import Levenshtein
import numpy as np
import pdb
import difflib

ref = "I love you too".split(' ')
hyp = "I really don't loie him".split(' ')

# return hypothesis wrong word with labels
def get_WER_highlight(ref, hyp):
    result = []
    for li in difflib.ndiff(ref, hyp):
        if li[0] == "+" or li[0] == " ":
            x = li.split(" ")
            # pdb.set_trace()
            if len(x) == 3:
                x = (x[-1], "1")
            else:
                x = (x[-1], "0")
            result.append(x)
    return result            


def diff_texts(text1, text2):
    d = difflib.Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]
    
# x = diff_texts(ref, hyp)

# pdb.set_trace()