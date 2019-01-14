import argparse
from create_transforms import create_transforms

parser = argparse.ArgumentParser(description='Word Image Similarity preprocessing script')

parser.add_argument("--data", type=str, default="../gw_20p_wannot", metavar="I",
                    help="directory for the input data")
parser.add_argument("--out", type=str, default="../preprocessed", metavar="O",
                    help="directory for the ouptut data")

args = parser.parse_args()

create_transforms(args)
