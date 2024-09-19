"""Simple parser for A3M."""

from typing import Union

from io import TextIOWrapper

from carabiner.cast import cast
import numpy as np

_A3M_ALPHABET = tuple("ARNDCQEGHILKMFPSTWYV-")
_A3M_ALPHABET_DICT = dict(zip(_A3M_ALPHABET, range(len(_A3M_ALPHABET))))

# read A3M and convert letters into
# integers in the 0..20 range,
def parse_a3m(filename: Union[str, TextIOWrapper]) -> np.ndarray:

    """Parse an A3M file into an array of token IDs.
    
    """

    msa = []
    for line in cast(filename, to=TextIOWrapper):  # read file line by line
        if line.startswith('>'):
            continue
        # remove lowercase letters and append to MSA
        msa.append([letter for letter in line.rstrip() if not letter.islower()])
    # convert letters into numbers
    msa = [[_A3M_ALPHABET_DICT.get(letter, len(_A3M_ALPHABET_DICT) - 1)  # default to gap if unknown letter
            for letter in line]
           for line in msa]
    return np.asarray(msa)

