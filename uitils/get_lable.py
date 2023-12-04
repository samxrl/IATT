import numpy as np
import os

def get_lable():
    lable = np.empty(shape=(0, 2), dtype=str)
    f = open('data/synset_words.txt', encoding='UTF-8')
    line = f.readline()
    data_list = []
    while line:
        data_list.append(line)
        line = f.readline()
    f.close()
    data_array = np.array(data_list)

    for i, line in enumerate(data_array):
        lable = np.append(lable, [[line[:9], line[10:-1]]], axis=0)

    return lable
