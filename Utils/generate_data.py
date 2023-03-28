import numpy as np
import argparse

parser = argparse.ArgumentParser(description="The parameters for generating the random number")
parser.add_argument("--row", "-r", default=100, type=int, help="Rows of the Matrix")
parser.add_argument("--col", "-c", default=100, type=int, help="Cols of the Matrix")
parser.add_argument("--range", "-R", default=100, type=int, help="Range of the number in Matrix")
parser.add_argument("--add", "-a", default=True, type=bool, help="Whether to add data or Overwrite it")
parser.add_argument("--filename", "-f", default="Player-Data/Input-P0-0", type=str, help="The location to save the matrix, .txt file is recommended")
args = parser.parse_args()


if args.add:
    file = open(args.filename, 'a')
else:
    file = open(args.filename, 'w')

tmp = np.random.randint(0,args.range,args.row*args.col)

for i in range(args.row * args.col):
    file.write(str(tmp[i])+" ")
    if i!=0 and i % args.row == 0:
        file.write("\n")
