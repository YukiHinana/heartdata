from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import shap
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
        # if user inputs an invalid file type
        # flash an error
        if not allowed_file(file.filename):
            flash('Invalid file type')
            return redirect(request.url)
        # if user does not input a format type
        # flash an error
        if 'hasIndex' not in request.form:
            flash('No selected index column option')
            return redirect(request.url)

        hasIndex = request.form['hasIndex']
        # if there are no errors redirect to confusion matrix
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if hasIndex == "True":
                return redirect(url_for('uploaded_file', filename=filename, hasIndex = True))
            else:
                return redirect(url_for('uploaded_file', filename=filename, hasIndex = False))
        
    return render_template('pigboi77.html')

@app.route('/uploads/index <hasIndex>/<filename>')
# on redirect read format and file
def uploaded_file(filename, hasIndex):
    data = pandas.read_csv(filename)
    cols = len(data.columns)
    #endCols = cols - 2
    #startRange = 0
    # set different column formatting rules based on selected option
    if hasIndex == "True":
        startRange = 1
        endCols = cols - 2
    else: 
        startRange = 0
        endCols = cols - 1

    # loads the dataset
    dataset = loadtxt(filename, delimiter=",", skiprows=1, usecols=range(startRange, cols))

    X = dataset[:,0:endCols]
    Y = dataset[:,endCols]

    seed = np.random.seed()
    # standard set_size at 20%
    test_size = .2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    # run model
    # model = XGBClassifier()
    # model = LGBMClassifier()
    # model = GaussianNB()
    # model = AdaBoostClassifier()
    # model = RandomForestClassifier()
    # model = CatBoostClassifier()
    models = [XGBClassifier(), LGBMClassifier(), GaussianNB(), AdaBoostClassifier(), RandomForestClassifier(), CatBoostClassifier()]
    mtables = []
    messages1m = []
    accuracies = []
    shaps1 = []
    index = 0
    for model in models:

        classType = ""

        if "CatBoostClassifier" in str(model):
            classType = "Cat Boost Classifier"
        elif "XGBClassifier" in str(model):
            classType = "XGB Classifier"
        elif "LGBMClassifier" in str(model):
            classType = "LGBM Classifier"
        elif "GaussianNB" in str(model):
            classType = "Gaussian NB Classifier"
        elif "AdaBoostClassifier" in str(model):
            classType = "Ada Boost Classifier"
        elif "RandomForestClassifier" in str(model):
            classType = "Random Forest Classifier"

        model.fit(X_train, Y_train)


        Y_pred = model.predict(X_test)
        predictions = [round(value) for value in Y_pred]
        # calculate and present accuracy
        accuracy = accuracy_score(Y_test, predictions)
        #flash("Classifier used is " + classType)
        messages1m.append("Classifier used is "+ classType + "\n")
        '''
        if accuracy == 1:
            flash("Accuracy: %.2f%%" % (accuracy * 100.0) + emoji.emojize(":hundred_points:"))
        elif accuracy >= .95:
            flash("Accuracy: %.2f%%" % (accuracy * 100.0) + emoji.emojize(":grinning_face:"))
        else:
            flash("Accuracy: %.2f%%" % (accuracy * 100.0) + emoji.emojize(":worried_face:"))
        '''
        if accuracy == 1:
            accuracies.append("Accuracy: %.2f%%" % (accuracy * 100.0) + emoji.emojize(":hundred_points:") + "\n")
        elif accuracy >= .95:
            accuracies.append("Accuracy: %.2f%%" % (accuracy * 100.0) + emoji.emojize(":grinning_face:") + "\n")
        else:
            accuracies.append("Accuracy: %.2f%%" % (accuracy * 100.0) + emoji.emojize(":worried_face:") + "\n")
        # create confusion matrix in dataframe
        confusion = confusion_matrix(Y_test, Y_pred)
        matrix = DataFrame({'Predicted 0': confusion[0],
                            'Predicted 1': confusion[1]},
                            index=['Actual 0', 'Actual 1'])

        mtables.append(matrix.to_html(classes='data'))

        shap_model = shap.TreeExplainer(model)
        shap_model_values = shap_model.shap_values(X_train)
        #   shap_df = DataFrame(shap_model_values, columns=X_train.columns.values)
        png_name = "summary_plot" + str(index)+".png"
        plot = shap.summary_plot(shap_model_values, X_train, show=false)
        plt.savefig(png_name)
        index += 1

    return render_template('confusionMatrix.html', tables=mtables, messages1=messages1m, accuracyList=accuracies)


if __name__ == "__main__":
  app.run(debug = True)

