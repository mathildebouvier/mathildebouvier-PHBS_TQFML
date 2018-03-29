
# coding: utf-8

# In[141]:


#Upload all the packages 
import numpy as np
import pandas as pd
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#Download the file
from sklearn import datasets
import csv

bank = pd.read_csv('/Users/Mathilde/Downloads/bank.csv', header = 0, delimiter=';')
bank_sub = pd.read_csv('/Users/Mathilde/Downloads/bank.csv', delimiter=';', skiprows=0, usecols=(0,5,6))


# In[49]:


bank


# In[87]:


#Convert string into float
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
bank_sub['marital']=class_le.fit_transform(bank['marital'].astype('str'))
bank_sub['job']=class_le.fit_transform(bank['job'].astype('str'))
bank_sub['education']=class_le.fit_transform(bank['education'].astype('str'))
bank_sub['default']=class_le.fit_transform(bank['default'].astype('str'))
bank_sub['housing']=class_le.fit_transform(bank['housing'].astype('str'))
bank_sub['loan']=class_le.fit_transform(bank['loan'].astype('str'))
bank_sub['contact']=class_le.fit_transform(bank['contact'].astype('str'))
bank_sub['month']=class_le.fit_transform(bank['month'].astype('str'))
bank_sub['poutcome']=class_le.fit_transform(bank['poutcome'].astype('str'))
bank_sub['y']=class_le.fit_transform(bank['y'].astype('str'))
bank_sub.head()


# In[52]:


bank.info()


# In[53]:


bank_sub.info()


# In[81]:


bank.describe().transpose()


# In[82]:


bank_sub.describe().transpose()


# In[99]:


#Separate the training/test data
from sklearn.model_selection import train_test_split

X, y = bank_sub.iloc[:, 1:].values, bank_sub.iloc[:, 0].values

X, X_test, y, y_test =train_test_split(X, y, test_size=0.3, random_state=0, stratify=Y)


# In[129]:


print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y))
print('Labels counts in y_test:', np.bincount(y_test))


# In[130]:


#Normalize
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_norm = mms.fit_transform(X)
X_test_norm = mms.transform(X_test)


# In[131]:


#Standardize
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X)
X_train_std = sc.transform(X)
X_test_std = sc.transform(X_test)


# In[133]:


X_train_std.shape


# In[134]:


#Logistic regression intuition and conditional probabilities 
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')

# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout()
#plt.savefig('images/03_02.png', dpi=300)
plt.show()


# In[135]:


#Learning the weights
def cost_1(z):
    return - np.log(sigmoid(z))


def cost_0(z):
    return - np.log(1 - sigmoid(z))

z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='J(w) if y=1')

c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')

plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.tight_layout()
#plt.savefig('images/03_04.png', dpi=300)
plt.show()


# In[136]:


#Logistic Regression Classifier using gradient descent
class LogisticRegressionGD(object):
    
    """Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state"""

    #Fit training data
    def fit(self, X, y):

        """Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object"""
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            
            # note that we compute the logistic `cost` now
            # instead of the sum of squared errors cost
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self
    
    #Calculate net input
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    #Compute logistic sigmoid activation
    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    #Return class label after unit step
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # equivalent to:
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


# In[116]:


X_train_01_subset = X[(y == 0) | (y == 1)]
y_train_01_subset = y[(y == 0) | (y == 1)]

lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)

plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)

plt.xlabel('x[standardized]')
plt.ylabel('y[standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/03_05.png', dpi=300)
plt.show()


# In[118]:


#SVM
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

bank(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel('x[standardized]')
plt.ylabel('y[standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_11.png', dpi=300)
plt.show()


# In[137]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X, y)

X_combined = np.vstack((X, X_test))
y_combined = np.hstack((y, y_test))
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree, test_idx=range(105, 150))

plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_20.png', dpi=300)
plt.show()


# In[138]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25, 
                                random_state=1,
                                n_jobs=2)
forest.fit(X, y)

plot_decision_regions(X_combined, y_combined, 
                      classifier=forest, test_idx=range(105, 150))

plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_22.png', dpi=300)
plt.show()


# In[139]:


#Check the accuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


# In[140]:


print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))

