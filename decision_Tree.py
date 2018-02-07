import monkdata as m
import dtree as tree
import random
import sys

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
        
Td,Vd = partition(m.monk1,0.8)    
calculate_best(Td,Vd)
