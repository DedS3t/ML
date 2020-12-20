# Implementation of standard log regression.

import numpy as np
import matplotlib.pyplot as plt

num_observations=5000
x1=np.random.multivariate_normal([0,0], [[1,.75],[.75,1]], num_observations)
x2=np.random.multivariate_normal([1,4], [[1,.75],[.75,1]], num_observations)

simulated_separableish_features=np.vstack((x1,x2)).astype(np.float32)
simulated_labels=np.hstack((np.zeros(num_observations),np.ones(num_observations)))

plt.figure(figsize=(12,8))
plt.scatter(simulated_separableish_features[:,0], simulated_separableish_features[:,1],c=simulated_labels,alpha=.4)
plt.show()

def sigmoid(scores):
    return 1/(1+np.exp(-scores))

#  log likelihood is calculated this formula, where y is the target class, xi is a data point and β is a weights vector ll=∑yiβTxi - log(1+e^(βTxi))
def log_likelihood(features,target,weights):
    scores=np.dot(features,weights)
    return np.sum(target*scores-np.log(1+np.exp(scores)))

def logistic_regression(features,target,num_steps,learning_rate,add_intercept=False):
    if add_intercept:
        intercept=np.ones((features.shape[0],1))
        features=np.hstack((intercept,features))
    weights=np.zeros(features.shape[1])

    for step in range(num_steps):
        scores=np.dot(features,weights)
        predictions=sigmoid(scores)

        # taking the derivative of the log likelihood equation(to maximize it) and changing to matrix form ∇ll=XT(Y-Predictions) 
        output_error_signal=target-predictions
        gradient=np.dot(features.T,output_error_signal)
        weights+=learning_rate*gradient
        if step%10000==0:
            print(log_likelihood(features, target, weights))
    return weights


weights=logistic_regression(simulated_separableish_features, simulated_labels,num_steps=30000, learning_rate=5e-5,add_intercept=True)
data_with_intercept=np.hstack((np.ones((simulated_separableish_features.shape[0], 1)),simulated_separableish_features))
final_scores=np.dot(data_with_intercept, weights)
preds=np.round(sigmoid(final_scores))
print('Accuracy from model: {}'.format((preds==simulated_labels).sum().astype(float)/len(preds)))

# if model was trained longer with a small lr it would eventually match the actual implementaions accuracy, cuz gradient ascent
# on a concave function will eventually reach global optimum
