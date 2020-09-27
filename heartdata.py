from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
import os
import emoji
from werkzeug.utils import secure_filename


basedir = os.path.abspath(os.path.dirname(__file__)) 

UPLOAD_FOLDER = os.path.join(basedir)
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('pigboi77.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
  dataset = loadtxt(filename, delimiter=",", skiprows=1, usecols=range(1,15))

  X = dataset[:,0:13]
  Y = dataset[:,13]

  seed = 7

  test_size = 0.2
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

  model = XGBClassifier()
  model.fit(X_train, Y_train)

  Y_pred = model.predict(X_test)
  predictions = [round(value) for value in Y_pred]

  accuracy = accuracy_score(Y_test, predictions)

  print("Accuracy: %.2f%%" % (accuracy * 100.0))

  confusion = confusion_matrix(Y_test, Y_pred)
  flash(str(confusion))

  return render_template('pigboi77.html')

if __name__ == "__main__":
  app.run(debug = True)

