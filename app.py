from flask import Flask
from flask import request
from flask import send_file

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils import _bhv_reg_df, _extract_fc
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting

app = Flask(__name__)

"""
TODO: For more reference and guide refer here: https://flask.palletsprojects.com/en/2.0.x/quickstart/#
"""

class Args:
    def __init__(self):
        self.input_data = 'data/'
        self.roi = 300
        self.net = 7
        self.bhv = 'ListSort_Unadj'
        self.zscore = 1
        self.correlation = 'correlation'
        self.corr_type= 'correlation'
        self.k_fold = 10
        self.corr_thresh = 0.2


"""
Main handler that makes prediction for a particular behaviour
TODO: Replace test datasets (x_test, y_test) with uploaded single-user dataset
TODO: Make prediction for a single-user
"""
@app.route("/predict/<string:behaviour>")
def get_prediction(behaviour):
    # 1. import our hdf5 best_model
    # 2. Make prediction using our model using model.predict() keras function
    # 3. return the metric values and predicted score

    # model = load_model(filename)
    # predictions = model.predict(x_test,verbose=0).squeeze()
    # # predictions = model.predict(xtest,verbose=0).squeeze()
    # mae  = mean_absolute_error(y_test, predictions)
    # mse  = mean_squared_error(y_test, predictions)
    # rmse = sqrt(mean_squared_error(y_test, predictions))
    # mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    # r2   = r2_score(y_test,predictions)
    # r, p = scipy.stats.spearmanr(predictions, y_test)

    # TODO: Replace mock data with actual metrics
    return {
        "mse": 12,
        "mae": 12,
        "correlation": 0.058,
        "epochs": 100,
        "predicted_score": 1
    }

@app.route('/upload')
def upload_dataset():
    return render_template('upload.html')

"""
Main handler that obtains the dataset uploaded by the user in front-end to the back-end API
TODO: refer to https://flask.palletsprojects.com/en/2.0.x/patterns/fileuploads/
TODO: alternatively can refer to https://pythonbasics.org/flask-upload-file/#:~:text=It%20is%20very%20simple%20to,it%20to%20the%20required%20location.
"""
@app.route("/uploader", methods = ['GET', 'POST'])
def upload_dataset():
    f = request.files['file']
    # f.save(secure_filename(f.filename))
    return 'Dataset uploaded!'

"""
Main handler that generates and returns the connectivity matrix for the uploaded dataset
"""
@app.route("/graphs/<string:behaviour>")
def show_graphs(behaviour):
    # import the uploaded dataset from data folder
    # use nilearn's graph library to plot our connectivity matrix
    # return the png image

    args = Args()
    bhv_data = _bhv_reg_df(args)    # load fmri data from file

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([bhv_data[0]['fmri']])[0]

    path = 'images/' + behaviour + '-conn-matrix.png'
    display = plotting.plot_matrix(correlation_matrix, colorbar=True, vmax=0.8, vmin=-0.8)
    display.figure.savefig(path)
    return send_file(path, mimetype='image/png')


"""
Main handler that returns the architecture of the deep learning model for a particular behaviour
TODO: Generate the architecture diagrams beforehand using online tool: 
"""
@app.route("/architecture/<string:behaviour>")
def show_architecture(behaviour):
    # TODO: replace test images with actual ones later
    return send_file('images/architecture/' + behaviour  + '-test.png', mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)