#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:12:16 2019

@author: combaldieu
"""
from shutil import copy2
import os


#ADAPTING THE PARZIVAL DATASET

source_file = "../parzivaldb-v1.0"
data_file = "../dataset_parzi/"
file_path = source_file + "/ground_truth/word_labels.txt"
if not os.path.exists('../dataset_parzi'):
    os.makedirs('../dataset_parzi')

words = ""
with open(file_path) as fp:  
   line = fp.readline()
   cnt = 1
   while line:
       pic_name, word = line.strip().split(" ")
       words += word + "\n"
       copy2(source_file + "/data/word_images_normalized/" + pic_name +".png",
             data_file + "word-{:06d}.png".format(cnt))
       
       print("Word {} copied : {}".format(cnt, line.strip()))
       
       line = fp.readline()
       cnt += 1


file = open(data_file + "words.txt", "w")
file.truncate(0)
file.write(words.strip())
file.close()

print("Dataset conversion complete !")