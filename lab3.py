#!/usr/bin/python
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

# ## Jupyter notebooks
# 
# In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.
# 
# If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.
# 
# And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.

# ## Import the libraries
# 
# In Jupyter, select the cell below and press `ctrl + enter` to import the needed libraries.
# Check out `labfuns.py` if you are interested in the details.

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random
import numpy.linalg as alg
import math as math
import xlwt
import sys
# ## Bayes classifier functions to implement
# 
# The lab descriptions state what each function should do.


# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    #find uniquel labels
    classes = np.unique(labels)
    #find the number of classes
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))
    #print("prior -> ", prior)
    # TODO: compute the values of prior for each class!
    # ==========================
    total = len(labels)
    k = []
    fin = []
    for i in classes:
        #clas.append(np.where(labels == i)[0])
        temp = len(np.where(labels == i)[0])
        k.append(temp)
        fin.append(k)
        k = []
    #print("k -> ", k)
    #print("fin -> ", fin)
    ret = np.array(fin).reshape(-1,1)
    rett = ret/total
    prior = rett
    #print("fin reshap -> ", ret)
    #print("fin div -> ", rett)
    
    # ==========================

    return prior

def get_specific(a,b):
    array = []
    for el in b:
        array.append(a[el])
    return array

def matrix_mult(B):
    A = B.reshape(-1,1)
    row = []
    mat = []
    for i in range(len(A)):
        row = []
        for j in range(len(B)):
            temp = A[i]*B[j]
            row.append(sum(temp))
        mat.append(row)
    return mat

# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    #assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    #print("unique? -> ", classes)
    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))
    #print("mu before -> ", mu)
    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    #calculation of mu
    clas = []
    specific =  []
    k = 0
    for i in classes:
        #clas.append(np.where(labels == i)[0])
        k = np.where(labels == i)[0]
        array = get_specific(X,k)
        clas.append(k)
        specific.append(array)
    u_means = []
    #print("Specific --------->>>>>> ", specific)
    #get number of features 
    #get number of features 
    nb = len(specific[0][0])
    for i in range(len(specific)):
        l = specific[i]
        tot = np.sum(l, axis= 0)
        tot = np.sum(l, axis= 0)/ len(specific[i])
        s = [x for x in tot]
        u_means.append(s)
    mu = u_means
    # ==========================
    #print("mu after -> ", mu)
    # ==========================
    #calculation of sigma

    #print("length of specific ", len(specific))
    sigm = []
    for i in range(len(specific)):
        s = [ x - u_means[i] for x in specific[i]]
        sigm.append(s)
    
    #print("State now -> ", sigm)
    
    sigms = []
    for i in range(len(sigm)):
        s = [ np.array(matrix_mult(x)) for x in sigm[i]]
        sigms.append(s)
    
    #print("State now -> ", sigms)
    
    total = 0
    sigmas = []
    for i in range(len(sigms)):
        long = len(sigms[i])
        sum = 0
        for j in range(long):
            sum = sum + sigms[i][j]
        sigmas.append(sum/long)
    #print("State now sigmas func -> ", sigmas)
    diag = [np.diag(x) for x in sigmas]
    sd = [np.diag(d) for d in diag]
    #print("Diag sigmas -> ", [np.diag(x) for x in sigmas] )
    #print("Diag sigmas mat -> ", sd)
    sigma = sd
    # ==========================
    return mu, sigma

####------------------------------ VERSION 2 -------------------------------------

# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams2(X, labels, W=None):
    #assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    #print("unique? -> ", classes)
    if W is None:
        W = np.ones((Npts,1))/float(Npts)
    
    #denomitor
    #print("W == ", W)
    #deno = np.sum(W)
    #print("deno (1) -> ", deno)
    #deno = 1 / deno
    #print("deno (2) --> ", deno)
    #end-
    
    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))
    #print("mu before -> ", mu)
    # TODO: fill in the code to compute mu and sigma!
    # ==========================
    #calculation of mu
    
    #u_num = 
    w_X = []
    for i in range(len(X)):
        ret = np.dot(X[i],np.sum(W[i]))
        w_X.append(ret)
    #print("Weighted X : --> : ", w_X)
    
    clas = []
    specific =  []
    u_specific =  []
    w_specific = []
    k = 0
    for i in classes:
        #clas.append(np.where(labels == i)[0])
        k = np.where(labels == i)[0]
        array_u = get_specific(w_X,k)
        array = get_specific(X,k)
        wspec = get_specific(W,k)
        clas.append(k)
        specific.append(array)
        w_specific.append(wspec)
        u_specific.append(array_u)
    u_means = []
    #print("class by index 0 : ", clas)
    #print("Weight Specific ---> ", w_specific)
    #print("Specific --------->>>>>> ", specific)
    #get number of features 
    #get number of features 
    nb = len(specific[0][0])
    for i in range(len(u_specific)):
        l = u_specific[i]
        deno_w = np.array(w_specific[i])
        deno_sum = np.sum(deno_w)
        #print("specific ", i, " ", l)
        tot = np.sum(l, axis= 0)
        #tot = [np.dot(x,) for x in l]
        #tot = np.sum(l, axis= 0)/ len(specific[i])
        tot = np.sum(l, axis= 0)/deno_sum
        s = [x for x in tot]
        u_means.append(s)
    mu = u_means
    # ==========================
    #print("mu after -> ", mu)
    # ==========================
    #CALCULATION OF SIGMA

    #print("length of specific ", len(specific))
    sigm = []
    for i in range(len(specific)):
        s = [ x - u_means[i] for x in specific[i]]
        sigm.append(s)
    
    #print("State now -> ", sigm)
    
    sigms = []
    for i in range(len(sigm)):
        s = [ np.array(matrix_mult(x)) for x in sigm[i]]
        sigms.append(s)
    
    #print("State now -> ", sigms)
    
    total = 0
    sigmas = []
    for i in range(len(sigms)):
        long = len(sigms[i])
        poids = w_specific[i]
        deno_w = np.array(poids)
        deno_sum = np.sum(deno_w)
        print("long len = ", long, " poids length =  ", len(poids) )
        sum = 0
        for j in range(long):
            sum = sum + poids[j]*sigms[i][j]
        sigmas.append(sum/deno_sum)
    #9*print("State now sigmas func -> ", sigmas)
    diag = [np.diag(x) for x in sigmas]
    sd = [np.diag(d) for d in diag]
    #print("Diag sigmas -> ", [np.diag(x) for x in sigmas] )
    #print("Diag sigmas mat -> ", sd)
    sigma = sd
    # ==========================
    return mu, sigma


# out: prior - C x 1 vector of class priors
def computePrior2(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    #find uniquel labels
    classes = np.unique(labels)
    #find the number of classes
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))
    #print("prior -> ", prior)
    # TODO: compute the values of prior for each class!
    # ==========================
    #total = len(labels)
    k = []
    fin = []
    for i in classes:
        #clas.append(np.where(labels == i)[0])
        #temp = len(np.where(labels == i)[0])
        index = np.where(labels == i)[0]
        temp = [ np.sum(W[i]) for i in index]
        k.append(np.sum(temp))
        fin.append(k)
        k = []
    #print("k -> ", k)
    #print("fin -> ", fin)
    ret = np.array(fin).reshape(-1,1)
    #normalization 
    total = np.sum(ret)
    rett = ret/total
    prior = rett
    #prior = ret
    #print("fin reshap -> ", ret)
    #print("fin div -> ", rett)
    
    # ==========================

    return prior



####----------------------------------------------------------------------------




#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#change 
#

X, labels = genBlobs(n_samples = 6 ,centers=2)
print("Data X ", X)
print("Class -> ", labels)
#X = [[0,4],[1,1],[3,3],[4,0],[4,0],[7,1],[8,4],[5,3]]
#X = np.array(X)
#labels = [0,0,0,0,1,1,1,1]


#TODO : JFFKFK +
#
#change 
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    # TODO: fill in the code to compute the log posterior logProb!
    # ==========================
    ln_pk = [math.trunc(np.log1p(sum(x))) for x in prior]
    ln_pk = np.array(ln_pk)
    ln_sigmak = [math.trunc((-1.0/2)*np.log1p(len(x)))  for x in sigma ]
    ln_sigmak = np.array(ln_sigmak)
    disc_func = 0
    demi = -1.0/2
    ss = []
    ssf = []
    for i in range(Npts):
        ss = []
        for j in range(Nclasses):
            x_star = X[i]
            u_k = mu[j]
            diff = x_star - u_k
            diff_t = diff.reshape(-1,1)
            invers = np.linalg.inv(sigma[j])
            delta = np.matmul(np.matmul(diff,invers),diff_t)
            #print("Delta BEFORE -> ", delta)
            delta = demi*sum(delta)
            #print("Delta AFTER -> ", delta)
            ss.append(math.trunc(delta))
        ssf.append(np.array(ss))
    # ==========================
    #print("classsybayes(1111) V x -> ", ssf)
    print("ln_pk ", ln_pk)
    #print("ln_sigmak ", ln_sigmak)
    #ssf = [ (y + np.array(ln_sigmak) + np.array(ln_pk)).tolist() for y in ssf]
    ssf = [ y + np.array(ln_sigmak) + np.array(ln_pk) for y in ssf]
    #print("classsybayes(2222) V x -> ", ssf)
    logProb = ssf
    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=1)
    return h


# The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:


# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# ## Test the Maximum Likelihood estimates
# 
# Call `genBlobs` and `plotGaussian` to verify your estimates.

######TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST

#X, labels = genBlobs(centers=5)
r = np.ones(6)
r = r*6
r = 1 / r
#r = r.reshape(-1,1)
#mu, sigma = mlParams(X,labels)
#print("mu 1 ", mu ," sigma 1 " , sigma)
mu, sigma = mlParams2(X,labels,r)
print("mu 2 ", mu ," sigma 2 " , sigma)

#plotGaussian(X,labels,mu,sigma)
rer = computePrior(labels)
print("comp prior (1) -> ", rer, " SUM UP TO (1) : ", np.sum(rer))
rer = computePrior2(labels, r)
print("comp prior (2) -> ", rer, " SUM UP TO (2) : ", np.sum(rer))
# print("mu mu : ", mu)
# print("sigmas -> ", sigma)
# Npts = X.shape[0]
# Nclasses,Ndims = np.shape(mu)
# print("Npts -> ", Npts)
# print("Nclasses -> ", Nclasses)
# ##classifyBayes---------------------------
# h = classifyBayes(X,rer,mu,sigma)
# print("The data : ", X)
# print("The labesl : ", labels)
# print("classified by bayes ? -> ", h)
Y = np.array([[1,5],[2,2]])
h_y = classifyBayes(Y,rer,mu,sigma)
print(Y," classified by bayes ? -> ", h_y)


# Call the `testClassifier` and `plotBoundary` functions for this part.

#print("Now 1")
#mean, std = testClassifier(BayesClassifier(), dataset='iris', split=0.7)
#print("Now 2")
#print("mean ", mean, " std ", std)

def print_xls(name,filename, datas):
    
    wb = xlwt.Workbook()
    ws = wb.add_sheet(name)
    ws.write(0,0,'split fraction')
    ws.write(0,1,' mean accuracy ')
    ws.write(0,2,' std ')
    
    i = 1
    split_array = [0.3,0.5,0.6,0.7,0.8]
    #datas = 'iris'
    for j in split_array:
        mean, std = testClassifier(BayesClassifier(), dataset=datas, split=j)
        ws.write(i,0,j) # Row, Column, Data.
        ws.write(i,1,mean)
        ws.write(i,2,std)
        i += 1
    wb.save(filename)


#print_xls('sheet 1','iris_data.xls','iris')
#print_xls('sheet 1','vowel_data.xls','vowel')

#testClassifier(BayesClassifier(), dataset='vowel', split=0.7)


#print("plot 1?")
#plotBoundary(BayesClassifier(), dataset='iris',split=0.7)
#print("plot 2222?")

# ## Boosting functions to implement
# 
# The lab descriptions state what each function should do.


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))


        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================================
        # ==========================================
        # do classification for each point
        # Here we return the predicted class labels by Bayes
        vote = classifiers[-1].classify(X)
        #Her we have the actual class labels
        actual = labels
        error, error_vec = computer_error(wCur,labels,vote)
        
        if error == 0.0:
            continue
        #very small number to make sure that we do not divide by 0
        delta = sys.float_info.epsilon
        #compute the alphas 
        alpha = (1.0/2)*(np.log( (1 - error) / (error + delta) ))
        
        #update the weights
        temp_weight = np.zeros(Npts)
        for j in range(len(error_vec)):
            if error_vec[j] == 1 :
                temp_weight[j] = wCur[j]*np.exp(-alpha)
            else:
                temp_weight[j] = wCur[j]*np.exp(alpha)
        
        #normalization of the weight 
        wCur = temp_weight/np.sum(temp_weight)
        alphas.append(alpha) # you will need to append the new alpha
        # ==========================
        print("Begin ------------------------------")
        print("The iteration number : ", i_iter)
        print("Actual ", actual)
        print("predicted class : ", vote)
        print(" error -> ", error)
        print("error vector -> ", error_vec)
        print("Alpha -> ", alpha)
        print("current Weigths -> ", wCur)
        print("Finish -----------------------------")
        
    return classifiers, alphas


def computer_error(W, true_val, predicted_val):
    error = 0
    error_vec = []
    
    for i in range(true_val):
        diff = true_val[i] != predicted_val[i]
        error = error + W[i]*diff
        error_vec.append(diff)
        
    return error, error_vec
    
    
        
# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        
        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
# 
# Call the `testClassifier` and `plotBoundary` functions for this part.


#testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)



#testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)



#plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)


# Now repeat the steps with a decision tree classifier.


#testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)



#testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)



#plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)



#plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)


# ## Bonus: Visualize faces classified using boosted decision trees
# 
# Note that this part of the assignment is completely voluntary! First, let's check how a boosted decision tree classifier performs on the olivetti data. Note that we need to reduce the dimension a bit using PCA, as the original dimension of the image vectors is `64 x 64 = 4096` elements.


#testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare this with using pure decision trees or a boosted bayes classifier. Not too bad, now let's try and classify a face as belonging to one of 40 persons!


#X,y,pcadim = fetchDataset('olivetti') # fetch the olivetti data
#xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) # split into training and testing
#pca = decomposition.PCA(n_components=20) # use PCA to reduce the dimension to 20
#pca.fit(xTr) # use training data to fit the transform
#xTrpca = pca.transform(xTr) # apply on training data
#xTepca = pca.transform(xTe) # apply on test data
# use our pre-defined decision tree classifier together with the implemented
# boosting to classify data points in the training data
#classifier = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr)
#yPr = classifier.classify(xTepca)
# choose a test point to visualize
#testind = random.randint(0, xTe.shape[0]-1)
# visualize the test point together with the training points used to train
# the class that the test point was classified to belong to
#visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])

