import chainer
from chainer import cuda,FunctionSet,Variable
import chainer.functions as F
import chainer.links as L

import numpy as np


def softmax(X):
    X = X.copy()
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X


def softmax_classification(t, y, classes):
    X_prob = softmax(t)
    predict_prob_indics = X_prob.argmax(axis=1)
    match = classes[predict_prob_indics] == y
    right = np.where(match == True)[0]
    wrong = np.where(match == False)[0]

    return right, wrong


#X = np.random.uniform(0, 1, (10, 5))
#y = np.random.randint(0, 5, 10)

#print(X)
#print(y)
#classes = np.array([0, 1, 2, 3, 4])
#print(softmax_cross_entropy2(X, y, classes))


class LeNet(chainer.Chain):

    def __init__(self):
        super(LeNet,self).__init__(

            conv1 = L.Convolution2D(1,32,5,stride = 1),
            #bn1   = F.BatchNormalization( 4),
            conv2 = L.Convolution2D(32,64,5,stride = 1),
            fc3 = L.Linear(1024,200),
            fc4 = L.Linear(200,11),

)


    def forward(self,x_data,y_data,train=True):
        x = Variable(x_data,volatile=not train)
        t = Variable(y_data,volatile=not train)
        h = F.max_pooling_2d(F.relu(self.conv1(x)),ksize=2,stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)),ksize=2,stride=2)
        h = F.dropout(F.relu(self.fc3(h)))
        h = self.fc4(h)

        return F.softmax_cross_entropy(h,t),F.accuracy(h,t)


    def predict(self,x_data,y_data,train=False):
        x = Variable(x_data,volatile=not train)
        t = Variable(y_data,volatile=not train)
        h = F.max_pooling_2d(F.relu(self.conv1(x)),ksize=2,stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)),ksize=2,stride=2)
        h = F.dropout(F.relu(self.fc3(h)))
        h = self.fc4(h)
        h.data=cuda.to_cpu(h.data)
        y_data=cuda.to_cpu(y_data)
        classes_ = np.arange(11)
        r_indices,w_indices = softmax_classification(h.data,y_data,classes_)
        return str(r_indices),str(w_indices)
        

