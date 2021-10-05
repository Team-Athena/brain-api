from flask import Flask
from flask import request
from flask import send_file
from flask import render_template
from flask import Response
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import os

import numpy as np
# from tensorflow.keras.models import load_model
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils import _bhv_reg_df, _extract_fc
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
import pandas as pd

UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'pkl'}


app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

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

def find_top_k_connections(FC,top_50=True,top_100=False):
     # use top-100 FC connections
    if top_100 or top_50:
        # FC(1:1+size(FC,1):end) = 0;%set diagonal to zeros
        rcID = np.argwhere( FC )
        rId, cId = rcID[:,0], rcID[:,1]
        if len(rId)>100 and top_50 == True: 
            A=sorted(FC.ravel(),reverse=True);
            k_pos = A[51];# top 50 (positive values)
            k_neg = A[-51];# top 50 (negative values)
            if k_neg>=0.0 and k_pos>0.0:
                FC[(FC<=k_pos)]=0;
            else:
                FC[(FC>=k_neg) & (FC<=k_pos)]=0;
        elif len(rId)>200 and top_100 == True:
            A=np.sort(FC.ravel(),'reverse');
            k_pos = A[101];# top 100 (positive values)
            k_neg = A[-101];#% top 100 (negative values)
            if k_neg>=0.0 and k_pos>0.0:
                FC[(FC<=k_pos)]=0;
            else:
                FC[(FC>=k_neg) & (FC<=k_pos)]=0;
        
    rcID = np.argwhere( FC!=0 ) ;# % find nonzero indices     
    return rcID

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

    args = Args()
    args.bhv = behaviour

    # model = load_model(filename)

    # predictions = model.predict(x_test,verbose=0).squeeze()
    # mae  = mean_absolute_error(y_test, predictions)
    # mse  = mean_squared_error(y_test, predictions)
    # rmse = sqrt(mean_squared_error(y_test, predictions))
    # mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    # r2   = r2_score(y_test,predictions)
    # r, p = scipy.stats.spearmanr(predictions, y_test)

    # TODO: Replace mock data with actual metrics
    return {
        "behavior": behaviour,
        "mse": 12,
        "mae": 12,
        "correlation": 0.058,
        "epochs": 100,
        "predicted_score": 1
    }

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

"""
Main handler that obtains the dataset uploaded by the user in front-end to the back-end API
TODO: refer to https://flask.palletsprojects.com/en/2.0.x/patterns/fileuploads/
TODO: alternatively can refer to https://pythonbasics.org/flask-upload-file/#:~:text=It%20is%20very%20simple%20to,it%20to%20the%20required%20location.
"""
@app.route("/upload", methods = ['GET', 'POST'])
def upload_dataset():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        if f and allowed_file(f.filename):
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            os.rename('data/' + filename, 'data/dataset.pkl')
            return Response('{ "message": "Dataset uploaded!" }', status=200, mimetype='application/json')
        else:
            return Response('{ "message": "Invalid dataset format!" }', status=400, mimetype='application/json')
    elif request.method == 'GET':
       return render_template('upload.html') 

"""
Main handler that generates and returns the connectivity matrix for the uploaded dataset
"""
@app.route("/graphs/<string:behaviour>")
def show_graphs(behaviour):
    # import the uploaded dataset from data folder
    # use nilearn's graph library to plot our connectivity matrix
    # return the png image

    power = pd.read_csv('coords/Schaefer2018_300Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv')
    coords = np.vstack((power['R'], power['A'], power['S'])).T

    # TODO: Implement caching
    args = Args()
    args.bhv = behaviour
    bhv_data = _bhv_reg_df(args)    # load fmri data from file

    path = 'images/' + behaviour + 'connectivity-matrix.png'
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([bhv_data[0]['fmri']])[0]
    # np.fill_diagonal(correlation_matrix, 0)
    # path = 'images/' + behaviour + '-conn-matrix.png'
    display = plotting.plot_matrix(correlation_matrix, colorbar=True, vmax=0.8, vmin=-0.8)
    display.figure.savefig(path)
    # return send_file(path, mimetype='image/png')
    return send_file(path, mimetype='image/png')


"""
Main handler that returns the architecture of the deep learning model for a particular behaviour
TODO: Generate the architecture diagrams beforehand using online tool: 
"""
# @app.route("/architecture/<string:behaviour>")
# def show_architecture(behaviour):
#     # TODO: replace test images with actual ones later
#     if (behaviour == 'working_memory'):
#         return send_file('images/architecture/working-memory-test.png', mimetype='image/png')
#     elif (behaviour == 'ListSort_Unadj'):
#         return send_file('images/architecture/working-memory-test.png', mimetype='image/png')
#     elif (behaviour == 'blabla'):
#         return send_file('images/architecture/working-memory-test.png', mimetype='image/png')
#     return send_file('images/architecture/' + behaviour  + '-test.png', mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)

