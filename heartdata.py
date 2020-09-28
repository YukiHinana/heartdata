from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import emoji
from pandas import *

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
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Invalid file type')
            return redirect(request.url)
        
        if 'hasIndex' not in request.form:
            flash('No selected index column option')
            return redirect(request.url)

        hasIndex = request.form['hasIndex']

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if hasIndex == "True":
                return redirect(url_for('uploaded_file', filename=filename, hasIndex = True))
            else:
                return redirect(url_for('uploaded_file', filename=filename, hasIndex = False))
        
    return render_template('pigboi77.html')

@app.route('/uploads/index <hasIndex>/<filename>')
def uploaded_file(filename, hasIndex):
    data = pandas.read_csv(filename)
    cols = len(data.columns)
    endCols = cols - 2
    startRange = 0
    if hasIndex == "True":
        startRange = 1
        endCols = cols - 2
    else: 
        startRange = 0
        endCols = cols - 1

    dataset = loadtxt(filename, delimiter=",", skiprows=1, usecols=range(startRange, cols))

    X = dataset[:,0:endCols]
    Y = dataset[:,endCols]

    seed = np.random.seed()

    test_size = .2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    model = XGBClassifier()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    predictions = [round(value) for value in Y_pred]

    accuracy = accuracy_score(Y_test, predictions)
    if accuracy == 1:
        flash("Accuracy: %.2f%%" % (accuracy * 100.0) + emoji.emojize(":hundred_points:"))
    elif accuracy >= .95:
        flash("Accuracy: %.2f%%" % (accuracy * 100.0) + emoji.emojize(":grinning_face:"))
    else:
        flash("Accuracy: %.2f%%" % (accuracy * 100.0) + emoji.emojize(":worried_face:"))

    confusion = confusion_matrix(Y_test, Y_pred)
    matrix = DataFrame({'Predicted 0': confusion[0],
                        'Predicted 1': confusion[1]},
                        index=['Actual 0', 'Actual 1'])


    return render_template('confusionMatrix.html',  tables=[matrix.to_html(classes='data')], titles=matrix.columns.values)

if __name__ == "__main__":
  app.run(debug = True)

