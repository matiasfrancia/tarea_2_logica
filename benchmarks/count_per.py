#!/usr/bin/env python
#-*- coding:utf-8 -*-
##

import inspect, os, sys
sys.path.insert(0, os.path.join(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])), '../pysat-module/'))
sys.path.insert(0, os.path.join(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])), '../hitman/py/'))


if __name__ == '__main__':
        # only one parameter: the path of the text results
    if not len(sys.argv) == 2:
        raise ValueError("It must give one exact parameter!!")
    
    dataset_path = sys.argv[1]
    
    pos_cnt = 0
    neg_cnt = 0

    with open(dataset_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = line.strip().split(',')[-1].strip()
            if label == '0':
                neg_cnt = neg_cnt + 1
            else:
                pos_cnt = pos_cnt + 1
    
    print("postive " + str(pos_cnt))
    print("negative " + str(neg_cnt))   
    print(pos_cnt * 1.0/ (neg_cnt+ pos_cnt))

