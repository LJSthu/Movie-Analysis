import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--kind', default = 0, type=int, help = 'which kind of recommender to use')
args = parser.parse_args()
