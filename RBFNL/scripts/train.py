"""Train an rbfln model

Usage:
  train.py -m <n> -n <n> -i <file> -o <file>
  train.py -h | --help

Options:
  -m <n>        Number of neurons in the hidden layer.
  -n <n>        Number of iterations.
  -i <file>     Input data file.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt

if __name__ == '__main__':
    opts = docopt(__doc__)
