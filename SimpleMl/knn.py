# KNN implementation from scratch
# Is the simplest ML model you can use, it is the most intuitive. Find the closest point


import numpy as np



def dist(x,y):
    return np.linalg.norm(x-y) # ||x-y||

# 1-NN simple example

# find closest point
def nearest_neighbor(x_new,trainX,trainY):
    idx=0
    closest_dist=float('inf')
    for i in range(len(trainX)):
        curr=dist(x_new,trainX[i])
        if curr<closest_dist:
            idx,closest_dist=i,curr
    return trainY[idx]

def k_nearest_neighbor(x_new,trainX,trainY,k=1):
  #find the closest point in train X
  diffrences=trainX-x_new
  distances=[np.linalg.norm(d) for d in diffrences]
  indices_of_nearest_neighbors=[]

  for i in range(k):
    indices_of_nearest_neighbors.append(np.argmin(distances))
    del distances[indices_of_nearest_neighbors[-1]]
    labels_of_nearest_neighbors=np.zeros(10)
    for i in range(k):
      labels_of_nearest_neighbors[trainY[indices_of_nearest_neighbors[i]]]+=1
      
  return np.argmax(labels_of_nearest_neighbors) #return label


from sklearn import datasets
# create training/test data
digits=datasets.load_digits()
digits_X=digits.data
digits_y=digits.target
indices=np.random.permutation(len(digits_X))
digits_X_train=digits_X[indices[0:1200]]
digits_y_train=digits_y[indices[0:1200]]
digits_X_test=digits_X[indices[1200:]]
digits_y_test=digits_y[indices[1200:]]

num_correct=0
for i in range(len(digits_X_test)):
  pred_label=nearest_neighbor(digits_X_test[i], digits_X_train, digits_y_train)
  if pred_label==digits_y_test[i]:
    num_correct+=1
print("1-NN got {}".format(num_correct/len(digits_X_test)))



