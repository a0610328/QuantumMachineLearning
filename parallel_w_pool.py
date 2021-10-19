# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 09:06:28 2021

@author: Huijie Guan

Pick More Daisies
"""

from multiprocessing import Pool, cpu_count
from time import sleep,time

def sleep_f(second):
    print(f'start sleeping for {second} seconds')
    sleep(second)
    return second

t = time()
with Pool(cpu_count()) as p:
    inputs = [1,2,3,4,5,6,7,8]
    results = p.map(sleep_f, inputs)
    
print(list(results))


