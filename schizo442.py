#Importing Libraries for Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('C:/Users/jesuloluwa/Desktop/juju/Schizophrenia17new.csv')
dataset.shape
dataset.describe
X = scaled_features.iloc[:, 0:16].values
y = scaled_features.iloc[:, 16].values

#Encoding Categorical Data using Label Encoder for respective columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])
labelencoder_X_5 = LabelEncoder()
X[:, 5] = labelencoder_X_5.fit_transform(X[:, 5])
labelencoder_X_6 = LabelEncoder()
X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])
labelencoder_X_10 = LabelEncoder()
X[:, 10] = labelencoder_X_10.fit_transform(X[:, 10])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)

#Feature Scaling (Standardization)
from sklearn.preprocessing import StandardScaler
scaled_features = dataset.copy()
col_names = ['DUR_EPIS']
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features[col_names] = features
print(scaled_features)

#Building of the CLASSISCHIZ 
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, AlphaDropout
from keras.callbacks import ModelCheckpoint

#Initializing the neural network classifier
classischiz = Sequential()

#Adding the input layer and the first hidden layer and the drop out nodes
classischiz.add(Dense(units = 8, kernel_initializer = 'random_uniform', activation = 'relu', input_dim = 16))
classischiz.add(Dropout(0.2))

#Adding the second hidden layer and the drop out nodes
classischiz.add(Dense(units = 4, kernel_initializer = 'random_uniform', activation = 'relu'))
classischiz.add(Dropout(0.2))

#Adding the output layer of the neural neywork classifier
classischiz.add(Dense(units = 1, kernel_initializer = 'random_uniform', activation = 'sigmoid'))

classischiz.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = classischiz.fit(X_train, y_train, validation_split = 0.2, batch_size = 16, epochs = 500)

scores = classischiz.evaluate(X_train, y_train, verbose = 0)
print("%s: %.2f%%" % (classischiz.metrics_names[1], scores[1]*100))

scores = classischiz.evaluate(X_test, y_test, verbose = 0)
print("%s: %.2f%%" % (classischiz.metrics_names[1], scores[1]*100))

#Model Architecture
classischiz.save("classischiz.h5")

classischiz = load_model('classischiz.h5')
classischiz.summary()

#Data Visualization
import seaborn as sns
plt.inline
import random as rnd

corr = dataset.corr()


k = sns.countplot(dataset["CLASS"])
for b in k.patches:
    k.annotate(format(b.get_height(),'.0f'),(b.get_x()+b.get_width() / 2.,b.get_height()))
    
#Data Visualization bef    
f, ax = plt.subplots(1, 2, figsize = (10, 7))
f.suptitle("Visualization of Data sets", fontsize = 15.)
_ = dataset.CLASS.value_counts().plot.bar(ax = ax[0], rot = 0, color = (sns.color_palette()[4], sns.color_palette()[2])).set(xticklabels = ["SCHIZ", "OTHERS"])
_ = dataset.CLASS.value_counts().plot.pie(labels = ("SCHIZ", "OTHERS"), autopct = "%.2f%%", label = "", fontsize = 13., ax = ax[1],\
colors = (sns.color_palette()[4], sns.color_palette()[2]), wedgeprops = {"linewidth": 1.5, "edgecolor": "#FFA07A"}), ax[1].texts[1].set_color("#FA8072"), ax[1].texts[3].set_color("#FA8072")


corr=dataset.corr()

sns.set(font_scale=1.15)
plt.figure(figsize=(10, 7))

sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='plasma',linecolor="blue")
plt.title('Correlation between features');




plt.figure(figsize=[7,5])
sns.heatmap(corr,annot= True)

y_pred = classischiz.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt = "d")


print(history.history.keys())

#Data Visualization 
#Model Accuracy Plot
plt.plot(history.history['acc'], color = 'purple')
plt.plot(history.history['val_acc'], color = 'green')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

#Model Loss Plot
plt.plot(history.history['loss'], color = 'purple')
plt.plot(history.history['val_loss'], color = 'green')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('scaler', StandardScaler(), ['DUR_EPIS'])], remainder = 'passthrough')
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

print(X_train)

X_train = ct.fit_transform(features)
X_test = sc.transform(X_test)

import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = classischiz.predict_proba(X_test)
preds = probs
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()








