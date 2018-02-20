import numpy , random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy import *
import matplotlib


A = numpy.concatenate((numpy.random.randn(10,2)*0.2 + [1.5 , 0.5], numpy.random.randn(10,2)*0.2 + [-1.5, 0.5]))

B = numpy.random.randn(20,2)*0.2 + [0.0, -0.5]

inputs = numpy.concatenate(( A,B ))
targets = numpy.concatenate(( numpy.ones(A.shape[0]), -numpy.ones(B.shape[0]) ))
N = inputs.shape[0] # Number of rows (samples )
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]
#print(inputs)

#print("ha -> ", inputs[1])
#print("ha -> ", inputs[1][1])

#print("A", A)
#print("B", B)
x = []
y = [] 
for i in inputs:
    x.append(i[0])
    y.append(i[1])
    
xA = []
yA = [] 
for i in A:
    xA.append(i[0])
    yA.append(i[1])
    #print("A0", i[0])
    #print("A1", i[1])

xB = []
yB = [] 
for i in B:
    xB.append(i[0])
    yB.append(i[1])

#plt.plot(x, y, 'r--', xA, yA, 'bs', xB, yB, 'g^')



#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(xA,yA, marker='s', s=90)
#ax.scatter(xB,yB, marker='o', s=50, c='red')
#plt.title('Support Vectors')

#plt.show()


#plt.plot(xB, yB, 'ro')
#plt.plot(xA, yA, 'bo')

def numbers_of_data(data):
    return len(data)

 
#data = [[1,1,1],[1,-1,-1],[-1,-1,1],[-1,1,-1]]
#data = [[1,2,-1],[-1,2,-1],[-1,-2,1], [3,1,1]]


data = [[3,1,1],[3,-1,1],[6,1,1],[6,-1,1],[1,0,-1],[0,1,-1],[0,-1,-1],[-1,0,-1]]
#data1 = [[3,1,1],[3,-1,1],[6,1,1],[6,-1,1],[1,0,1],[0,1,1],[0,-1,1],[-1,0,1]]
#data = [[1,2,-1],[-1,2,-1],[-1,-2,1]]

def calculate_coeffs(data):
    coefs_u = []
    coefs_kernel = []
    a_u = []
    b_k = []
    cf_u = []
    cf_k = []
    index = []
    x1 = []
    x2 = []
    s = ""
    for i in range(len(data)):
        for j in range(len(data)):
            #print("i ", i, " j ", j)
            coefs_u.append(data[i][2]*data[j][2])
            a_u.append(data[i][2]*data[j][2])
            #
            x1.append(data[i][0])
            x1.append(data[i][1])
            #x1.append(data1[i][2])
            #
            x2.append(data[j][0])
            x2.append(data[j][1])
            #x2.append(data1[j][2])
            #
            prov = linear_kernel(x1,x2)
            #
            coefs_kernel.append(prov)
            b_k.append(prov)
            s = str(i) + str(j)
            index.append(s)
            del x1[:]
            del x2[:]
        #print("here a_u-> ", a_u)
        #print("here b_k-> ", b_k)
        cf_u.append(a_u)
        cf_k.append(b_k)
        a_u = []
        b_k = []
    #print("coef_u  ", coefs_u, " len ", len(coefs_u))
    #print("coef_k  ", coefs_kernel, " len ", len(coefs_kernel))
    #print("s_index ", index, " len ", len(index))
    #print("coef_u_fin ", cf_u)
    #print("coef_k_fin ", cf_k)
    return cf_u, cf_k

def linear_kernel(x1,x2):
    #return math.pow(x1[0]*x2[0] + x1[1]*x2[1] + 1, 2)
    return numpy.dot(x1,x2)
    #return math.pow(numpy.dot(x1,x2) + 1, 2)

def cal(n):
    for i in range(n):
        for j in range(n):
            print("i = ",i , " j ", j)

def iterations(n):
    return (n*(n-1)) / 2
def coefficients(data):
    # 0 <= i <= n - 2
    # 1 <= j <= n - 1
    l = len(data)
    diag = []
    not_diag = []
    not_diag_index = []
    #multiply mat and kernel elementwise
    row = []
    mat = []
    iter = iterations(len(data))
    unit, kernel = calculate_coeffs(data)
    for i in range(len(data)):
        for j in range(len(data)):
            row.append(unit[i][j]*kernel[i][j])
        mat.append(row)
        row = []
    unit, kernel = calculate_coeffs(data) 
    #print("Mat K by unit ",mat)
    #until here ok
    for i in range(len(data)):
        for j in range(len(data)):         
            if i != j :
                if (0 <= i <= l - 2 ) and ( 1 <= j <= l - 1 ) and (i < j):
                    #print("i ", i, " j ", j)
                    not_diag.append(mat[i][j] + mat[j][i])
                    not_diag_index.append(i)
                    not_diag_index.append(j)
            else :
                diag.append(mat[i][j])


    return diag, not_diag, not_diag_index

diag,not_diag, index = coefficients(data)

#
# Here we are trying to calculate 
#  sum of alphas(i) from 1<i< nb_data
# 
def part1_alphas(alphas, data):
    nb_data = numbers_of_data(data)
    sum = 0
    for i in range(nb_data):
        #print("apphas index = ", i)
        sum = sum + alphas[i]
    
    return sum

def part2_alphas(alphas, data):
    frac = -1/2
    diag,not_diag, index = coefficients(data)
    len_diag = len(diag)
    diag_tot = 0
    len_not_diag = len(not_diag)
    not_diag_tot = 0
    for i in range(len_diag):
        diag_tot = diag_tot + alphas[i]*alphas[i]*diag[i]
    
    # minus - 1 baby
    diagonal_not = []
    r1 = 0
    r2 = 1
    #print("index is !-> ", index)
    for i in range(len_not_diag):
        #print("a_1 ", r1, " a_2 ", r2)
        a_1 = index[r1]
        a_2 = index[r2]
        r1 = r1 + 2
        r2 = r2 + 2
        
        diagonal_not.append(alphas[a_1]*alphas[a_2])
    #print("not diag mult -> ",diagonal_not)
    
    for i in range(len_not_diag):
        not_diag_tot = not_diag_tot + diagonal_not[i]*not_diag[i]    
    #print("diagonal sum -> ", diag_tot)
    return frac*(not_diag_tot + diag_tot)
        
def objective(x, d = data, sign = -1.0):
    return sign*(part1_alphas(x,d) + part2_alphas(x,d))

def constraint1(x, d = data):
    cons = 0
    #l = []
    for i in range(len(data)):
       # print("i = ", i)
        cons = cons + x[i]*data[i][2]
        #l.append(x[i]*data[i][2])
    #print("constraint -> ",l)
    return cons

def constraint2(x, d = data):
    cons = 0
    #l = []
    for i in range(len(data)):
        for j in range(len(data)):
           # print("cons1 --> i ", i, " j ", j)
            cons = cons + x[i]*data[j][2]
    return cons

def find_wo(w,index_s = 0):
    vec = [data[index_s][0], data[index_s][1] ]
    t = linear_kernel(w,vec)
    t = -t + ((1/data[index_s][2]))
    return t

def boundary(y = 1000):
    return [0,y]
def bounds(C,d = data):
    b = boundary(C)
    list = []
    for i in range(len(data)):
        list.append(b)
    #create a list of tuples
    temp = [tuple(x) for x in list]
    #create a tuple of tuples
    fin = tuple(temp)
    return fin   
cons1 = {'type':'eq','fun':constraint1}
cons = [cons1]
x0 = [1,2,1,1,1,5,1,1]
#x0 = [1,1,1,1]
bnds = bounds(1000)
sol = minimize(objective, x0, method = 'SLSQP', bounds = bnds, constraints=cons)
sol2 = minimize(objective, x0, bounds = bnds, constraints=cons)

print("diag -> ", diag)
print("not_diag -> ", not_diag)
print("index -> ", [x + 1 for x in index])
#a,b = objective([0,1/8,1/8],data)
#print(a)
#print(b)
#print("obj 1111 ??? -> ", objective([0,1/8,1/8],data))
#print("obj 2222 ? -> ", objective([0,1/8,1/8]))
#print("cons1 " , constraint1([0,1/8,1/8]))
print("bounds", bounds(5))
#print("sol -> ", sol.x)
#k = [  2.42861287e-17, 0, 1.38777878e-17]
#print("cons sol test " , constraint1(k))
#print("obj reallllly ??? -> ", objective(k,data))
print("sol -> ", sol.x)
print("sol 2 -> ", sol2.x)
print("LENGTH sol 2 -> ", len(sol2.x))


array = []
result = [] 
summ = 0
for i in range(len(sol.x)):

    alpha = sol.x[i]
    array.append(data[i][0])
    array.append(data[i][1])
    dot_product = linear_kernel(data[i][2]*alpha, array)
    summ = summ + dot_product
    result.append(dot_product)
    array = []
summ = summ.tolist()
summ.append(find_wo(summ))
print("The result is : ", result )
print("The fin result is : ", summ )
#print("wo ??? -> ", find_wo(summ))

def decision_boundary(w):
    if math.floor(round(w[0])) == 0 :
        return 'y',-math.floor(round(w[2]))/math.floor(round(w[1],2))
    elif math.floor(round(w[1])) == 0 :
        return 'x',-round(w[2])/round(w[0],2)
    else :
        return 'xy',-round(w[0],2)/round(w[1],2),round(w[2],2)/round(w[1],2)

print(decision_boundary(summ))

x_line = arange(-2.0, 4.0, 0.1)
y_line = arange(-2.0, 4.0, 0.1)
print("length ----> ",len(x_line))
xx_a = []
yy_a = []
xx_b = []
yy_b = []
for i in range(len(data)):
    if data[i][2] == 1 :
        xx_a.append(data[i][0])
        yy_a.append(data[i][1])
    else:
        xx_b.append(data[i][0])
        yy_b.append(data[i][1])    

#
decision = decision_boundary(summ)
x_l = 0
y_l = 0
if decision[0] == 'x':
    y_l = x_line
    x_l = [decision[1] for x in range(len(x_line))]
if decision[0] == 'y':
    x_l = y_line
    y_l = [decision[2] for x in range(len(x_line))]
if decision[0] == 'xy':
    x_l = x_line
    y_l = [decision[1]*x + decision[2] for x in x_line]
#

plt.figure(1)
plt.subplot(211)
plt.plot(xx_a, yy_a, 'bo', xx_b, yy_b, 'r^',x_l, y_l, 'g')
    
plt.subplot(212)
plt.plot(x_l, y_l, 'r')
plt.show()

