
# coding: utf-8

# Jiajun Du CS5661 Homework FACE Project

# In[2]:


import numpy as np
import pandas as pd

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics


# In[3]:


get_ipython().magic('matplotlib inline')


# # #A Read image and plot

# In[4]:


i=0


# In[5]:


a = mpimg.imread('/Users/thp/Documents/CSULA/5661 Data Science/hw3/HW3/Face/'+str(i)+'.jpg')


# In[6]:


a


# In[8]:


a.size


# In[12]:


label.head()


# In[13]:


#ploting the image
plt.imshow(a, cmap=plt.cm.gray_r, interpolation='nearest')


# # #B build feature matrix

# In[14]:


a[0]


# In[16]:


a[0].size


# In[17]:


col = np.append(a[0],a[1])


# In[18]:


col


# In[19]:


col.size


# In[22]:


#create one vector for one image
for x in range (2,64):
    col = np.append(col,a[x])


# In[23]:


col.size


# In[28]:


#creating a dataFrame for the images
images = pd.DataFrame(data=col, columns=['0'])


# In[34]:


images.size


# In[35]:


#use a loop to include all images as different columns into above dataFrame

for i in range(1,400):
    b = mpimg.imread('/Users/thp/Documents/CSULA/5661 Data Science/hw3/HW3/Face/'+str(i)+'.jpg')
    col = np.append(b[0],b[1])
    for x in range (2,64):
        col = np.append(col,b[x])
    images[str(i)]=col
    
images.head()


# In[38]:


#make the feature matrix cooridnate with the dimensions of the label 
feature = images.T

feature.head()


# # #C Normalize feature matrix

# In[87]:


from sklearn import preprocessing


# In[88]:


feature_scaled = preprocessing.scale(df)


# In[90]:


feature_scaled


# # #D split the into train and test data set

# In[53]:


label = pd.read_csv('/Users/thp/Documents/CSULA/5661 Data Science/hw3/HW3/Face/label.csv')


# In[55]:


label.head()


# In[57]:


label.size


# In[91]:


#spliting data accroding to the assignment
X_train, X_test, y_train, y_test = train_test_split(feature_scaled, label, test_size=0.25, random_state=5)


# # #E use PCA to do dimentionality reduction

# In[59]:


from sklearn.decomposition import PCA


# In[60]:


k=50


# In[92]:


my_pca = PCA(n_components = k)
X_train_new = my_pca.fit_transform(X_train)
X_test_new = my_pca.transform(X_test)


# In[126]:


X_train_new.shape


# In[127]:


X_test_new.shape


# In[168]:


y_new = pd.concat([y_train,y_test])


# In[169]:


y_new.shape


# # #F Use non-linear SVM classifier 

# In[75]:


from sklearn.svm import SVC


# In[96]:


my_SVM = SVC(C=1,kernel='rbf',gamma=0.0005,random_state=1)


# In[103]:


my_SVM.fit (X_train_new, y_train)


# In[98]:


y_predict = my_SVM.predict (X_test_new)


# In[113]:


#check accuracy

score_SVM= accuracy_score(y_test, y_predict)
print(score_SVM)


# In[125]:


#Confusion Matrix

cm_SVM = metrics.confusion_matrix(y_test, y_predict)
print("Confusion matrix:")
print(cm_SVM)


# # #Grid Search for best C

# In[129]:


#merging X dataset
X_1 = pd.DataFrame(X_train_new)
X_1.head()


# In[130]:


X_2 = pd.DataFrame(X_test_new)
X_2.head()


# In[172]:


X_new = pd.concat([X_1,X_2])


# In[145]:


X_new_df = pd.concat(X_new)
X_new_df.shape


# In[ ]:


#Grid Search 


# In[146]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# In[160]:


param_grid = {
    'kernel':('linear','rbf'),
    'C':[0.1, 1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5]
    
}


# In[161]:


grid = GridSearchCV(my_SVM, param_grid,cv=10,scoring ='accuracy')


# In[170]:


grid.fit(X_new_df, y_new)


# In[171]:


print(grid.best_score_)
print(grid.best_params_)

