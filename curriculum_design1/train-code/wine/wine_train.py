import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import DataPreprocessing
from model.naive_bayes_classifier import NaiveBayesClassifier
from model.network import Network
from model.random_forest import RandomForest
import warnings

warnings.filterwarnings('ignore')

print("Dataset:wine")
data_x, data_y = DataPreprocessing()
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

print("naive bayes classifier:")
model_naive_bayes_classifier = NaiveBayesClassifier()
model_naive_bayes_classifier.train(X_train, y_train)
model_naive_bayes_classifier.predict(X_test, y_test)

model_sklearn_naive_bayes_classifier = GaussianNB()
model_sklearn_naive_bayes_classifier.fit(X_train, y_train)
y_pred_naive_bayes_classifier = model_sklearn_naive_bayes_classifier.predict(X_test)
print("sklearn model accuracy:", np.sum(y_pred_naive_bayes_classifier == y_test) / len(y_test))

print("network:")
model_network = Network(X_train.shape[1], 10, np.unique(y_train).size, alpha=0.05)
model_network.train(X_train, y_train)
model_network.predict(X_test, y_test)

model_sklearn_network = MLPClassifier()
model_sklearn_network.fit(X_train, y_train)
y_pred_network = model_sklearn_network.predict(X_test)
print("sklearn network accuracy:", np.sum(y_pred_network == y_test) / len(y_test))

print("random forest:")
model_random_forest = RandomForest()
model_random_forest.train(X_train, y_train)
y_pred = model_random_forest.predict(X_test, y_test)

model_sklearn_random_forest = RandomForestClassifier()
model_sklearn_random_forest.fit(X_train, y_train)
y_pred_random_forest = model_sklearn_random_forest.predict(X_test)
print("sklearn random forest accuracy:", np.sum(y_pred_random_forest == y_test) / len(y_test))