from preprocessing import preprocessing
from register_pairs import register_pairs
import argparse

parser = argparse.ArgumentParser(description='Word Image Similarity preprocessing script')

parser.add_argument("--input", type=str, default="../gw_20p_wannot", metavar="I",
                    help="directory for the input data")
parser.add_argument("--out", type=str, default="../preprocessed", metavar="O",
                    help="directory for the ouptut data")
parser.add_argument("--preprocessed", type=bool, default=True)

args = parser.parse_args()

if not args.preprocessed:
    preprocessing(args)
register_pairs()
