from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from flask import Flask, render_template, flash
import emoji

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

@app.route('/')
def machine_learning_model():
  dataset = loadtxt('heart_data_resampled.csv', delimiter=",", skiprows=1, usecols=range(1,15))

  X = dataset[:,0:13]
  Y = dataset[:,13]

  seed = 1
  test_size = 0.2
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

  model = XGBClassifier()
  model.fit(X_train, Y_train)

  Y_pred = model.predict(X_test)
  predictions = [round(value) for value in Y_pred]

  accuracy = accuracy_score(Y_test, predictions)

  print("Accuracy: %.2f%%" % (accuracy * 100.0))

  confusion = confusion_matrix(Y_test, Y_pred)

  flash (str(confusion[0]) + "\n" + str(confusion[1]))
  return render_template('pigboi77.html')

if __name__ == "__main__":
    app.run(debug = True)

