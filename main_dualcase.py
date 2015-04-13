# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:10:16 2015

@author: gavrilio
"""

import os
import network_dualcase
import numpy as np
import pandas as pd  
import csv
import matplotlib.pyplot as plt 
from scipy import stats   

def normalize(vec):
    max_ = np.max(vec)
    min_ = np.min(vec)
    vec = 2*(vec - min_)/float((max_ - min_))-1
    return vec,max_,min_
    
def normalize_predictioninput(vec,dfmax_,dfmin_):
    max_ = dfmax_
    min_ = dfmin_
    vec = 2*(vec - min_)/float((max_ - min_))-1
    return vec

def normalize_ybnd(vec,max_,min_,ynormalisationFunction):
    max_ = max_
    min_ = min_
    if ynormalisationFunction != "minus_one_one":
        vec = (vec - min_)/float((max_ - min_)) #---better for tanh
    else:
        vec = 2*(vec - min_)/float((max_ - min_))-1 #--better for sigmoid
    return vec

def unnormalize_ybnd(vec,max_,min_,ynormalisationFunction):
    max_ = max_#+abs(max_)
    min_ = min_#-abs(min_)
    if ynormalisationFunction != "minus_one_one":
        vec = (float((max_ - min_))*vec+min_)
    else:
        vec = (float((max_ - min_))*(vec+1)/2+min_)
    return vec


def vectorized_result(response,numOutputNodes):
    """Return a 2-dimensional unit vector for a
    0-1 response of logistic regression."""
    e = np.zeros((numOutputNodes, 1))
    if numOutputNodes > 1:
        e[response] = 1.0
        return e
    else:
        e[0] = response
        return response
      
def calculate_MSE(pred,actual):
    sum_ = 0
    sum__ = 0
    for i in range(len(pred)):
        sum_ += (pred[i] - actual[i])**2
        sum__ += abs(pred[i] - actual[i])
    
    mse = sum_/float(len(pred))
    acc = 1-sum__/float(len(pred))
    return (mse,acc)
    
def csv_writer(data, path):

    """
    Write data to a CSV file path
    """
 
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for row in data:
            writer.writerow([row])
    return   
    
    
    
 
def randomSampleSelections(df,trainingProportion):
    randVar = np.random.uniform(0,1,1)
    training_start = int(np.ceil(randVar*len(df)))     
    training_skip = len(df)/(len(df)*(1-trainingProportion))

    df["selection"] = ""

    selection_counter = 1
    training_skip_increment = 1
    df['selection'].iloc[training_start] = "validation"
    for i in range(training_start+1,len(df.index)+training_start):
        if i >= len(df.index):
            j = i - len(df.index)
        else:
            j = i
        if selection_counter % int(np.floor(training_skip*training_skip_increment)) == 0:
            df['selection'].iloc[j] = "validation"
            training_skip_increment +=1 
        else:
            df["selection"].iloc[j] = "training"
        selection_counter += 1
    
    

def createDataMatrix(df):
    depVar = np.array(df[df.columns[0]])
    indepVars = np.array(df[df.columns[1]])
    for i in range(2,len(df.columns)):
        if df.columns[i] != "selection":
            var =  np.array(df[df.columns[i]])
            indepVars = np.vstack((indepVars, var))
    indepVars = np.transpose(indepVars)
    return indepVars,depVar
 
def createSampleDataSets_TrainingValidationEtc(df,indepVars,depVar,numOutputNodes):
    numIndepVars = len(indepVars[0])
    validation_inputs = []
    validation_results = []
    training_inputs = [] 
    training_results = []
    trg_inputs = []
    trg_results = []
    pred_input = []
    pred_result = []

    for i in range(len(df)):
        if df["selection"].iloc[i] == "validation":
            validation_inputs.append(np.reshape(indepVars[i], (numIndepVars, 1)))
            validation_results.append(depVar[i])   
        else: 
            training_inputs.append(np.reshape(indepVars[i], (numIndepVars, 1)))
            training_results.append(vectorized_result(depVar[i],numOutputNodes))
 
    for i in range(len(df)):
            trg_inputs.append(np.reshape(indepVars[i], (numIndepVars, 1)))
            trg_results.append(depVar[i])       
            pred_input.append(np.reshape(indepVars[i], (numIndepVars, 1)))
            pred_result.append(depVar[i])
    trg_data = zip(trg_inputs,trg_results)       
    training_data = zip(training_inputs, training_results)
    validation_data = zip(validation_inputs,validation_results)
    return (training_data, validation_data, trg_data, pred_input,pred_result,numIndepVars)


def readInData(fn,discreteCase,activationFunction,transformationFunction,overSample):
    dfin = pd.read_csv(fn,sep=",",header=False)
    headers = [ col for col in dfin.columns]
    order = [True for col in dfin.columns]
#    dfin = dfin.sort(headers,ascending=order)
    print dfin.index
    if overSample == "True":
        oversamp = []
#        means = dfin.mean(axis=1)
#        print means
#        for i in range(0,len(dfin.index)):
#            if means[i] < 0.5:
        for i in range(0,len(dfin.index)):
            if dfin[dfin.columns[1]][i] == 1 or dfin[dfin.columns[1]][i] == 5:
                if np.random.randint(0,2) == 1:
                    oversamp.append(i)
            else:
                oversamp.append(i)
        print len(oversamp)
        df = dfin.iloc[oversamp]
    else:
        df = dfin

# taking bootstrap samples
    
#    for i in range(1,4):
#        sample = np.random.choice(df.index, 61878)
#        print sample
#        df = df.append(dfin.iloc[sample])

# sorting data
    df = df.sort(headers,ascending=order)

# adding extra columns    
#    countij = 0
#    for i in range(1,len(df.columns)):
#        for j in range(i+1,min(i+5,len(df.columns))):
#            countij = countij+1
#            print countij
#            df['new'+str(countij)] = np.multiply(df[df.columns[i]],df[df.columns[j]])
        
    print len(df.columns)
    dfmax = np.zeros((len(df.columns)-1,1))
    dfmin = np.zeros((len(df.columns)-1,1))
    dy = np.array(df[df.columns[0]])
    if not discreteCase:
        df[df.columns[0]] = normalize_ybnd(np.array(df[df.columns[0]]),np.max(df[df.columns[0]]),np.min(df[df.columns[0]]),ynormalisationFunction)
    for i in range(1,len(df.columns)):
        if df.columns[i] != "selection":
            if transformationFunction == "bin":
                df[df.columns[i]] =[(x if x < 31 else 50 ) for x in df[df.columns[i]]]
            elif transformationFunction == "binlog":
                df[df.columns[i]] =[(0.5 if x==0 else (x if x < 31 else 50) ) for x in df[df.columns[i]]]
                df[df.columns[i]] = np.log(df[df.columns[i]])
            elif transformationFunction == "log":
                df[df.columns[i]] =[(0.5 if x==0 else x) for x in df[df.columns[i]]]
                df[df.columns[i]] = np.log(df[df.columns[i]])
            elif transformationFunction == "sqrt":
                df[df.columns[i]] = np.sqrt(df[df.columns[i]])
            elif transformationFunction == "boxcox":
                df[df.columns[i]] = stats.boxcox(np.array(df[df.columns[i]]))[0]
            df[df.columns[i]],dfmax[i-1],dfmin[i-1] = normalize(np.array(df[df.columns[i]]))
            
    return df,dy,dfmax,dfmin

def readIn_PredictionData(fn,dfmax,dfmin,transformationFunction):
    df = pd.read_csv(fn,sep=",",header=False)

#    countij = 0
#    for i in range(0,len(df.columns)):
#        for j in range(i+1,min(i+5,len(df.columns))):
#            countij = countij+1
#            df['new'+str(countij)] = np.multiply(df[df.columns[i]],df[df.columns[j]])
        
    print len(df.columns)
    for i in range(0,len(df.columns)):
#        if df.columns[i] != "selection":
        if transformationFunction == "bin":
            df[df.columns[i]] =[(x if x < 31 else 50 ) for x in df[df.columns[i]]]
        elif transformationFunction == "binlog":
            df[df.columns[i]] =[(0.5 if x==0 else (x if x < 31 else 50) ) for x in df[df.columns[i]]]
            df[df.columns[i]] = np.log(df[df.columns[i]])
        elif transformationFunction == "log":
            df[df.columns[i]] =[(0.5 if x==0 else x) for x in df[df.columns[i]]]
            df[df.columns[i]] = np.log(df[df.columns[i]])
        elif transformationFunction == "sqrt":
            df[df.columns[i]] = np.sqrt(df[df.columns[i]])
        elif transformationFunction == "boxcox":
            df[df.columns[i]] = stats.boxcox(np.array(df[df.columns[i]]))[0]
        df[df.columns[i]] = normalize_predictioninput(np.array(df[df.columns[i]]),dfmax[i],dfmin[i])
    return df

def createPredictionDataMatrix(df):
    indepVars = np.array(df[df.columns[0]])
    for i in range(1,len(df.columns)):
        if df.columns[i] != "selection":
            var =  np.array(df[df.columns[i]])
            indepVars = np.vstack((indepVars, var))
    indepVars = np.transpose(indepVars)
    
    numIndepVars = len(indepVars[0])
    pred_input = []
    for i in range(len(df)):
            pred_input.append(np.reshape(indepVars[i], (numIndepVars, 1)))
    return pred_input


def determine_number_of_output_nodes(df):
    v = np.array(df[df.columns[0]])
    v = np.array(v)
    if(len(np.unique(v)) < 0.5*len(v)):
        return len(np.unique(v))
    else:
        return 1
    
    
def plot_linear_case(predicted,originalResponseVar,fn,activationFunction):    
    fig = plt.figure()
    ax = plt.subplot(111) 
    predicted = np.array([elem[0][0] for elem in predicted])
    predicted = unnormalize_ybnd(predicted,np.max(originalResponseVar),np.min(originalResponseVar),activationFunction)
    ax.plot(originalResponseVar,originalResponseVar)
    ax.scatter(originalResponseVar,predicted,c='b')
    mse,acc = calculate_MSE(predicted,originalResponseVar)
    title = "goodness of fit %.4f" %mse
    fig.suptitle(title)
    plt.savefig(fn.split(os.sep)[-1].split(".")[0]+".jpg")
    
   
if __name__ == "__main__":
    #-----INPUTS-------
    #fn = ".."+os.sep+"data"+os.sep+"titanic_bin_train_classgender.csv"
    fn = "D:"+os.sep+"kaggle"+os.sep+"otto"+os.sep+"data"+os.sep+"otto_train_num.csv"
    #fn = ".."+os.sep+"data"+os.sep+"example5.csv"
    discreteCase = True
    ynormalisationFunction = "zero_one" # "minus_one_one" or "zero_one"  choice is active for non discrete case
    activationFunction = "sigmoid" #"sigmoid" or "tanh"
    outputactivationFunction = "softmax" # activationFunction, "linear", "sigmoid" or "tanh"
    transformationFunction = "log" # none, bin, binlog, log (0 -> 0.5 edit), sqrt, boxcox   
    overSample = "False" #  "False, "True"

    if activationFunction == "tanh":
        ActivationFunction = network_dualcase.TanhActivation
    else:
        ActivationFunction = network_dualcase.SigmoidActivation

    if outputactivationFunction == "tanh":
        OutputActivationFunction = network_dualcase.TanhActivation
    elif outputactivationFunction == "sigmoid":
        OutputActivationFunction = network_dualcase.SigmoidActivation
    elif outputactivationFunction == "linear":
        OutputActivationFunction = network_dualcase.LinearActivation
    elif outputactivationFunction == "softmax":
        OutputActivationFunction = network_dualcase.SoftMaxActivation
        
    
    trainingProportion=0.9
    #when classification True, when continuous False
    df,originalResponseVar,dfmax,dfmin = readInData(fn,discreteCase,activationFunction,transformationFunction,overSample)
    numOutputNodes =  determine_number_of_output_nodes(df)
    randomSampleSelections(df,trainingProportion)
    indepVars,depVar = createDataMatrix(df)
    training_data, validation_data, trg_data, pred_input,pred_result,numVariables \
    =createSampleDataSets_TrainingValidationEtc(df,indepVars,depVar,numOutputNodes)

            
    # 100 & 6 best score 80/90 for validation
    numNodesInHiddenLayer = 50
    #numNodesInHiddenLayer2 = 0
    net = network_dualcase.Network([numVariables, numNodesInHiddenLayer,30, numOutputNodes], 
                                   cost=network_dualcase.LogLossCost,
                                   mainActivationFunction=ActivationFunction,
                                   mainOutputActivationFunction=OutputActivationFunction)

    numEpochs = 3
    numMiniBatches = 20
    LearningRate = 0.1
    Lmbda = 5 # regularization
    model = net.SGD(training_data, numEpochs, numMiniBatches, LearningRate,numOutputNodes,
                    lmbda= Lmbda, 
            evaluation_data=validation_data,monitor_evaluation_accuracy=True, 
            monitor_training_accuracy=True,monitor_training_cost=True,monitor_evaluation_cost=True)

    df_wgts_output = pd.DataFrame(model[4][1])
#    print df_wgts_output
    df_bias_output = pd.DataFrame(model[5][1])
#    print df_bias_output
    
    predicted,fulloutput = net.predict(pred_input,numOutputNodes)
    reality = pred_result
    MSE, ACC =  calculate_MSE(predicted,reality)
    print "MSE & accuracy of whole training ="
    print MSE, ACC

    a = predicted
#print a
    with open("D:/kaggle/otto/results/otto_fit.csv", 'wb') as outcsv:   
    #configure writer to write standard csv file
        writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(['number'])
        for item in a:
        #Write item to outcsv
            writer.writerow([item])

    b = reality
#print a
    with open("D:/kaggle/otto/results/otto_trg.csv", 'wb') as outcsv:   
    #configure writer to write standard csv file
        writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(['number'])
        for item in b:
        #Write item to outcsv
            writer.writerow([item])



    if not discreteCase:
        plot_linear_case(predicted,originalResponseVar,fn,ynormalisationFunction)


    prediction_input = readIn_PredictionData("D:"+os.sep+"kaggle"+os.sep+"otto"+os.sep+"data"+os.sep+"otto_test_num.csv",dfmax,dfmin,transformationFunction)
    prediction_datamatrix = createPredictionDataMatrix(prediction_input)
    predicted_newfile = net.predict(prediction_datamatrix,numOutputNodes)
    print "predicted survived ", sum(predicted_newfile[0])


    a = predicted_newfile[1]  # for softmax 0-8 for sigmoid 0-1

#print a
    with open("D:/kaggle/otto/results/otto_pred.csv", 'wb') as outcsv:   
    #configure writer to write standard csv file
        writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number', 'number'])
        for item in a:
        #Write item to outcsv
            writer.writerow([item[0][0], item[1][0], item[2][0], item[3][0], item[4][0], item[5][0], item[6][0], item[7][0], item[8][0]])
    outcsv.close
#        writer.writerow([item[0][0], item[1][0]]) # for separate models


    #fig,ax = plt.subplots(nrows=1,ncols=1)
    #fig.subplots_adjust(hspace=0.2)
    if discreteCase == False:
            fig = plt.figure()
            ax = plt.subplot(111) 
            predicted = np.array([elem[0][0] for elem in predicted])
            predicted = unnormalize_ybnd(predicted,np.max(dy),np.min(dy))
            real_dy = dy
            residuals = np.subtract(predicted,real_dy)
            ax.scatter(real_dy,real_dy,c='r')
            ax.scatter(real_dy,predicted)
            plt.show()
    #fig,ax = plt.subplots(nrows=1,ncols=1)
    #fig.subplots_adjust(hspace=0.2)
    else:
            fig = plt.figure()
            ax = plt.subplot(111) 
            pred_dy = np.array(predicted)
            real_dy = reality
            residuals = np.subtract(pred_dy,real_dy)
            ax.scatter(real_dy,real_dy,c='r')
            ax.scatter(real_dy,pred_dy)
            plt.show()


# another print method slow with [ & ] as content 
#with open("D:/kaggle/otto/data/otto_pred.csv", 'wb') as f:
#    csv.writer(f).writerows(np.array(predicted_newfile[1]))


#titanic example        
#    csv_writer(predicted,"../data/titanic_pred.csv")


#    sum_=0
#    for i in range(len(validation_data)):
#        sum_+= validation_data[i][-1]
#    print "number of survived in validation data set:",sum_
        
