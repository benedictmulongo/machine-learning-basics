import monkdata as m
import dtree as tree
import random
import sys
import matplotlib.pyplot as plt 
import numpy as np 

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def calculate_best(Td,Vd):

    error = -sys.maxsize
    counter = 0
    current_tree = tree.buildTree(Td,m.attributes)
    tr = tree.buildTree(Td,m.attributes)
    tr_pruned = tree.allPruned(tr)
    
    while True:
        counter = 0
        count = len(tr_pruned)
        
        for x in tr_pruned:
            if tree.check(x,Vd) > error:
                error = tree.check(x,Vd)
                current_tree = x
                print("current tree")
                print(current_tree)
                print("error")
                print(error)
            else:
                counter = counter + 1
        
        if count == counter:
            break
            
        tr = current_tree
    
    print("Selected tree:")
    print(tr)
    print("error:")
    print(error)
    return error, tr
    
def find_max_pos(lst):
    i = 0
    length = len(lst)
    current_max = lst[0]
    index = 0
    for x in range(length):
        if current_max <= lst[x] :
            current_max = lst[x]
            index = x
    return current_max, index      
                 
def calculate_bestavg(rep = 10, fraction = 0.7):
    
    all_test_errors = []
    all_test_trees = [] 
    x_axis = []
    for i in range(0,rep):
        Td,Vd = partition(m.monk3,fraction)   
        error, tr = calculate_best(Td,Vd)
        all_test_errors.append(error)
        all_test_trees.append(tr)
        x_axis.append(i+1)
    
    print(all_test_errors)
    print(all_test_trees)
    max, index = find_max_pos(all_test_errors)
    print("Max : " , max)
    print("Index : ", index)
    plt.title('Monk1 best performance for 100 samples') 
    plt.ylabel('percentage of corrected classification') 
    plt.xlabel('samples')
    plt.bar(x_axis,all_test_errors)
    plt.show()
    
     
calculate_bestavg(rep = 5)






 
