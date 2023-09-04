# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from eda import perform_eda
from preprocess import preprocess
import seaborn as sns



#Loading the dataset
data = pd.read_csv('./OnlineNewsPopularity.csv')
print("Head of the data:\n", data.head())

print("Performing EDA:\n")
perform_eda(data)

print("\nPreprocessing the data:\n")
x_train,x_test,y_train,y_test= preprocess(data)

# Fitting classification algorithms 

def plot_confusion_matrix(cm):
  # Create a heatmap for the confusion matrix
  sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")

  # Add labels, title, and axis ticks
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.xticks(ticks=[0, 1], labels=['Not popular', 'popular'])
  plt.yticks(ticks=[0, 1], labels=['Not popular', 'popular'])

  # Show the plot
  plt.show()
  
label_encoder = preprocessing.LabelEncoder()
ytrain_label = pd.Series(label_encoder.fit_transform(y_train>=1800)).to_frame()
ytest_label = pd.Series(label_encoder.fit_transform(y_test>=1800)).to_frame()
  
  
# Decision Tree

from sklearn.tree import DecisionTreeClassifier
    
dec_classifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
    
dec_classifier.fit(x_train,ytrain_label)
    
#calculating the accuracy
y_pred_dec_class = dec_classifier.predict(x_test)
cm = confusion_matrix(ytest_label,y_pred_dec_class)
plot_confusion_matrix(cm)
print("\nAccuracy score of decision tree classifier:\n",accuracy_score(ytest_label,y_pred_dec_class)*100)
    
#saving the model
save_classifier = open("./pickled_algos/dectreeclassifier.pickle","wb")
pickle.dump(dec_classifier, save_classifier)
save_classifier.close()


"""**Random Forest**"""

from sklearn.ensemble import RandomForestClassifier

ran_classifier = RandomForestClassifier(n_estimators = 50,criterion = 'entropy', random_state = 0)
ran_classifier.fit(x_train,ytrain_label)

#calculating the accuracy
y_pred_ran_class = ran_classifier.predict(x_test)
cm = confusion_matrix(ytest_label,y_pred_ran_class)
plot_confusion_matrix(cm)
print("\nAccuracy score of random forest classifier:\n",accuracy_score(ytest_label,y_pred_dec_class)*100)

#saving the model
save_classifier = open("./pickled_algos/ranforestclassifier.pickle","wb")
pickle.dump(ran_classifier, save_classifier)
save_classifier.close()

"""**ADA Boost**"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

# create an instance of the decision tree classifier
dec_tree_classifier = DecisionTreeClassifier(max_depth=1)

# create an instance of the adaboost classifier
ada_classifier = AdaBoostClassifier(
    base_estimator=dec_tree_classifier,
    n_estimators=50,
    learning_rate=1.0,
    algorithm='SAMME.R',
    random_state=0
)

# fit the classifier to the training data
ada_classifier.fit(x_train, y_train)

# predict on the test data
y_pred_ada = ada_classifier.predict(x_test)

# calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred_ada)
print(f"Accuracy score of ADABoost classifier: {accuracy*100:.2f}%")

# save the trained model to a file
with open('./pickled_algos/ada_boost_classifier.pickle', 'wb') as f:
    pickle.dump(ada_classifier, f)


"""**Naive Bayes classifier**"""

# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, confusion_matrix
# import pickle

# # Creating the classifier
# naive_classifier = GaussianNB()

# # Training the classifier
# naive_classifier.fit(x_train, y_train.values.ravel())

# # Predicting on test data
# y_pred_naive_class = naive_classifier.predict(x_test)

# # Evaluating the performance
# cm = confusion_matrix(y_test, y_pred_naive_class)
# plot_confusion_matrix(cm)
# print("\nAccuracy score of Naive Bayes classifier:\n",accuracy_score(y_test, y_pred_naive_class)*100)

# # Saving the model
# save_classifier = open("./pickled_algos/naivebayes.pickle", "wb")
# pickle.dump(naive_classifier, save_classifier)
# save_classifier.close()


"""**ANN fitting**"""

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.metrics import mean_absolute_error

ann_classifier = Sequential()

#Input layer
ann_classifier.add(Dense(units=15, kernel_initializer='uniform', input_dim=x_train.shape[1], activation='relu'))

# Hidden layers
ann_classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))
ann_classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))
ann_classifier.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))

# Output layer
ann_classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))


# Compile the model
ann_classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
ann_classifier.summary()

ann_classifier.fit(x_train, ytrain_label, epochs=10, batch_size=10)

#calculating the accuracy
y_pred_ann_class = ann_classifier.predict(x_test)
y_pred_ann_class = (y_pred_ann_class > 0.5)
cm = confusion_matrix(ytest_label,y_pred_ann_class)
plot_confusion_matrix(cm)

print("\nAccuracy score of Ann classifier\n",accuracy_score(ytest_label,y_pred_ann_class)*100)

#saving the model
save_classifier = open("./pickled_algos/annclassifier.pickle","wb")
pickle.dump(ann_classifier, save_classifier)
save_classifier.close()


"""# Fitting Regression Algorithms """

def calculate_model_performance(y_test, y_pred):
  # Calculate Mean Squared Error (MSE)
  mse = mean_squared_error(y_test, y_pred)

  # Calculate Root Mean Squared Error (RMSE)
  rmse = np.sqrt(mse)

  # Calculate Mean Absolute Error (MAE)
  mae = mean_absolute_error(y_test, y_pred)

  # Calculate R-squared (R2)
  r2 = r2_score(y_test, y_pred)

  print("Mean Squared Error (MSE):", mse)
  print("Root Mean Squared Error (RMSE):", rmse)
  print("Mean Absolute Error (MAE):", mae)
  print("R-squared (R2):", r2)
  
"""Linear regression"""
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

#calculating the accuracy
y_pred = regressor.predict(x_test)

calculate_model_performance(y_test, y_pred)

y_pred_df = pd.DataFrame(y_pred)
y_test.index = range(y_test.size)
result = pd.concat([y_test, y_pred_df], axis=1)

#saving the model
save_classifier = open("./pickled_algos/linearregression.pickle","wb")
pickle.dump(regressor, save_classifier)
save_classifier.close()

"""Decision Tree Regression"""

from sklearn.tree import DecisionTreeRegressor

dec_regressor = DecisionTreeRegressor(random_state=0)
dec_regressor.fit(x_train,y_train.values)

#calculating the accuracy
y_pred_dec = dec_regressor.predict(x_test)
calculate_model_performance(y_test, y_pred)
y_frame = pd.DataFrame(y_pred_dec)
result = pd.concat([y_test, y_frame], axis=1)

#saving the model
save_classifier = open("./pickled_algos/dectreeregression.pickle","wb")
pickle.dump(dec_regressor, save_classifier)
save_classifier.close()


"""Random Forest Regression""" 

from sklearn.ensemble import RandomForestRegressor

ran_regressor = RandomForestRegressor(n_estimators=15 , random_state=0)
ran_regressor.fit(x_train,y_train.values)

#calculating the accuracy
y_pred_ran = ran_regressor.predict(x_test)
calculate_model_performance(y_test, y_pred_ran)
y_ran_frame = pd.DataFrame(y_pred_ran)
result1 = pd.concat([y_test, y_ran_frame], axis=1)
result1

#saving the model
save_classifier = open("./pickled_algos/randomForestregression.pickle","wb")
pickle.dump(ran_regressor, save_classifier)
save_classifier.close()

"""**ANN Fitting**"""

from keras.models import Sequential

ann_regressor = Sequential()

# Input layer
ann_regressor.add(Dense(units=15, kernel_initializer='uniform', input_dim=x_train.shape[1], activation='relu'))

# Hidden layers
ann_regressor.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))
ann_regressor.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))
ann_regressor.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))

# Output layer
ann_regressor.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))

# Compile the model
ann_regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

# Print model summary
ann_regressor.summary()

ann_regressor.fit(x_train, y_train, epochs=10, batch_size=10)

y_pred_ann_regre = ann_classifier.predict(x_test)

calculate_model_performance(y_test, y_pred_ann_regre)

#saving the model
save_classifier = open("./pickled_algos/annregression.pickle","wb")
pickle.dump(ann_regressor, save_classifier)
save_classifier.close()