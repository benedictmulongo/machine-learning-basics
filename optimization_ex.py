
# min x1x4( x1 + x2 + x3) + x3 objective function
# s.t x1x2x3x4 >= 25  cons 1
# x1^2 + x2^2 + x3^2 + x4^2 = 40 cons 2
# 1 <=x1,x1,x1,x1 <= 5  bounds
# x0 = (1,5,5,1)   initial guess
#
#

import numpy as np 
from scipy.optimize import minimize

def objective(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    
    return x1*x4*(x1 + x2 + x3) + x3
    
def constraint1(x):
    return x[0]*x[1]*x[2]*x[3] - 25.0

def constraint2(x):
    sum_sq = 40
    for i in range(4):
        sum_sq = sum_sq - x[i]**2
    return sum_sq
    
#initial guess
start = np.zeros(4)
#make use of randomization we choosing start values 
start2 = [3,3,3,4]
x0 = [1,5,5,1]

print(objective(x0))
print(objective(start))

#bouns
b = (1.0,5.0)
bnds = (b,b,b,b)
cons1 = {'type':'ineq','fun':constraint1}
cons2 = {'type':'eq','fun':constraint2}
cons = [cons1,cons2]

#solution

sol = minimize(objective, x0, method = 'SLSQP', bounds = bnds, constraints=cons)
print(sol)
print(sol.fun)
print(sol.x)



sol2 = minimize(objective, start2, method = 'SLSQP', bounds = bnds, constraints=cons)

print(sol2.x)
