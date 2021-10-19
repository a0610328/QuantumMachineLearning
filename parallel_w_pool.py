# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 09:23:26 2021

@author: Huijie Guan

Pick More Daisies
"""

import time
from multiprocessing import Pool, cpu_count

def sleep_func(second):
    print(f'sleep for {second} second')
    time.sleep(second)
    return second

def func(n):
    return n*n

if __name__ =="__main__":
# =============================================================================
#     with Pool(cpu_count()) as p:
#         inputs = [1,2,3,4,5,6]
#         results = p.map_async(sleep_func, inputs)
# =============================================================================
    t1 = time.time()
    array = [1,2,3,4]*4
    p = Pool()
    result = p.map(sleep_func, array)
    p.close()
    p.join()
    
    print(f'time used is {time.time()-t1}')
    #collected_results = list(results)