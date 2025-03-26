

import concurrent.futures
import random
import string
import gzip
import json
import random
import re
import string
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
import numpy as np



def random_id(length=4):
    characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
    random_id = ''.join(random.choices(characters, k=length))
    return random_id


def file_to_string(filepath):
    with open(filepath, 'r') as f:
        data = f.read().strip()
    return data


def list_to_string(list_2d):
    sublists_as_strings = [f"[{','.join(map(str, sublist))}]" for sublist in list_2d]
    return f"[{','.join(sublists_as_strings)}]"



