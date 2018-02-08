import monkdata as m
import dtree as tree
import random
import sys
import matplotlib.pyplot as plt 
import numpy as np 
import drawtree_qt4 as draw
import matplotlib as mpl

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
                #print("current tree")
                #print(current_tree)
                #print("error")
                #print(error)
            else:
                counter = counter + 1
        
        if count == counter:
            break
            
        tr = current_tree
    
   # print("Selected tree:")
    #print(tr)
    #print("error:")
    #print(error)
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
                 
def calculate_bestavg(rep = 10, fraction = 0.7,filename = 'plot123.png', data = m.monk3):
    
    all_test_errors = []
    all_test_trees = [] 
    x_axis = []
    sum = 0

    average = 0
    for i in range(0,rep):
        Td,Vd = partition(data,fraction)   
        error, tr = calculate_best(Td,Vd)
        sum = sum + error
        all_test_errors.append(error)
        all_test_trees.append(tr)
        x_axis.append(i+1)
        
    average = sum /rep
    #standard deviation
    variance  = np.var(all_test_errors)
    std_error = np.sqrt(np.var(all_test_errors))
    print("The variance : ", variance)
    print("The standard deviation error  : " , std_error)
    
    #f-----------------
    #print(all_test_errors)
    #print(all_test_trees)
    max, index = find_max_pos(all_test_errors)
   # print("Max : " , max)
    #print("Index : ", index)
    #print("Tree max : ", all_test_trees[index])
    #print("The average is ", average)
    plt.title('Monk1 best performance for 100 samples') 
    plt.ylabel('percentage of corrected classification') 
    plt.xlabel('samples')
    plt.bar(x_axis,all_test_errors)
    
    line1_x = []
    line1_y = []
    line2_x = []
    line2_y = []
    
    for i in range(0,rep+2):
        line1_x.append(i)
        line1_y.append(average + std_error)
        line2_x.append(i)
        line2_y.append(average - std_error)
    
    plt.plot(line1_x,line1_y, 'r--')
    plt.plot(line2_x,line2_y, 'g--')
     # plt.savefig(filename)
    
    plt.show()
    print(variance)
    return max, all_test_trees[index], average


def calculate_diff_fractions(img = 'plot.png' , dat = m.monk3 , nb = 3):
    fraction = 0
    x_axis = []
    y_axis = []
    count = 1
    plt.bar(x_axis,y_axis)
    for i in range(3,9):
        fr = round(i*0.1,2)
        max, ttr, avg= calculate_bestavg(rep = 10, fraction = fr, data = dat)
        #y_axis.append(max)
        #x_axis.append(count)
        plt.bar(count,max, color = 'red')
        plt.text(count,max,'Max')
        #y_axis.append(avg)
        #x_axis.append(count+0.5)
        plt.bar(count+0.8,avg, color = 'green')
        plt.text(count+0.8,avg,'average')
        plt.text(count+0.5,0,'fract '+ str(fr))
        print("FOR FRACTION : ",fr)
        print("Max : " , max)
        print("Tree max : ", ttr)
        print("The average is ", avg)
        count = count + 3
    plt.ylabel('Percentage of corrected classification') 
    plt.xlabel('Fractions')
    plt.title('Monk ' + str(nb) + ' best performance for 10 samples') 
    plt.savefig(img)
    plt.close()
   # plt.show()

 
 
max,trr, avge = calculate_bestavg(fraction = 0.5, rep = 10, data = m.monk3)
draw.drawTree(trr)
 
#calculate_diff_fractions()
#for i in range(0,10):
#    calculate_diff_fractions(img = 'plotmonk3_'+ str(i) + 'png', dat = m.monk3, nb = 3 )
#for i in range(0,10):
#    calculate_diff_fractions(img = 'plotmonk1_'+ str(i) + 'png', dat = m.monk3, nb = 1 )



#drw_test = tree.buildTree(m.monk1,m.attributes)
#draw.drawTree(drw_test)

#0.3; 0.4; 0.5; 0.6; 0.7; 0.8




 
