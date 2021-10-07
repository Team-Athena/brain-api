from flask import Flask
from flask import request
from flask import send_file
from flask import render_template
from flask import Response
from flask.templating import render_template_string
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import os
import pickle
import pandas as pd

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils import _bhv_reg_df, _extract_fc, _info
from math import sqrt
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from nilearn import datasets

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
        # self.input_data = 'data/'
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
    np.fill_diagonal(FC, 0)
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
            A=sorted(FC.ravel(), reverse=True);
            k_pos = A[101];# top 100 (positive values)
            k_neg = A[-101];#% low 100 (negative values)
            if k_neg>=0.0 and k_pos>0.0:
                FC[(FC<=k_pos)]=0;
            else:
                FC[(FC>=k_neg) & (FC<=k_pos)]=0;
        print(k_pos)
        print('negatives', k_neg)

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
   
    # args = Args()
    # args.bhv = behaviour

    # _info(args.bhv)

    # bhv_data = _bhv_reg_df(args)
    # bhv_data = bhv_data[0]
    # fc_data,labels, IDs = _extract_fc(dataset, args.corr_type)

    with open('data/dataset.pkl', 'rb') as file:
        dataset = pickle.load(file)

    conn_measure = ConnectivityMeasure(kind='correlation')
    connectivity = conn_measure.fit_transform([dataset.T])[0]

    # subj_list = np.unique(IDs)
    # split_ration = int(0.8*len(np.unique(IDs)))
    
    # x = []
    # for i in range(split_ration,len(subj_list)):
    #     x.append(fc_data[np.where(IDs==subj_list[i])[0],...])
    # x = np.concatenate(x,0)

    # x = x[...,None]

    # if behaviour = working memory, get the best model for that behaviour
    if behaviour == "ListSort_Unadj":
        model = load_model('data/best_model_working_memory.hdf5')
        # print(connectivity[None,...,None].shape)
        predictions = model.predict(connectivity[None,...,None],verbose=0)
        # print(predictions[0][0])
        # predictions = list(predictions)
        # print(predictions)

        return {
        "behavior": behaviour,
        "mse": 0.02,
        "mae": 0.12,
        "correlation": 0.011,
        "epochs": 100,
        "predicted_score": str(predictions[0][0])
        }

    elif behaviour == "ProcSpeed_Unadj":
        model = load_model('data/best_model_processing_speed.hdf5')
        predictions = model.predict(connectivity[None,...,None],verbose=0)
        # predictions = list(predictions)
        return {
        "behavior": behaviour,
        "mse": 0.03,
        "mae": 0.15,
        "correlation": 0.019,
        "epochs": 100,
        "predicted_score": str(predictions[0][0])
        }

    elif behaviour == "PMAT24_A_CR":
        model = load_model('data/best_model_fluid_intelligence.hdf5')
        predictions = model.predict(connectivity[None,...,None],verbose=0)
        # predictions = list(predictions)
        return {
        "behavior": behaviour,
        "mse": 0.04,
        "mae": 0.16,
        "correlation": 0.02,
        "epochs": 100,
        "predicted_score": str(predictions[0][0])
        }




    # TODO: Replace mock data with actual metrics
    # return {
    #     "behavior": behaviour,
    #     "mse": 12,
    #     "mae": 12,
    #     "correlation": 0.058,
    #     "epochs": 100,
    #     "predicted_score": 1
    # }

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
    with open('data/dataset.pkl', 'rb') as file:
        dataset = pickle.load(file)

    # power = pd.read_csv('coords/Schaefer2018_300Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv')
    # coords = np.vstack((power['R'], power['A'], power['S'])).T

    # TODO: Implement caching
    args = Args()
    args.bhv = behaviour
    # bhv_data = _bhv_reg_df(args)    # load fmri data from file

    path = 'images/' + behaviour + 'connectivity-matrix.png'
    # correlation_measure = ConnectivityMeasure(kind='correlation')
    # correlation_matrix = correlation_measure.fit_transform([bhv_data[0]['fmri']])[0]
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([dataset.T])[0]
    # np.fill_diagonal(correlation_matrix, 0)
    # path = 'images/' + behaviour + '-conn-matrix.png'
    display = plotting.plot_matrix(correlation_matrix, colorbar=True, vmax=0.8, vmin=-0.8)
    display.figure.savefig(path)

    return send_file(path, mimetype='image/png')

@app.route("/3d-graph/<string:behaviour>")
def show_3d_graph(behaviour):

    with open('data/dataset.pkl', 'rb') as file:
        dataset = pickle.load(file)

    args = Args()
    args.bhv = behaviour
    # bhv_data = _bhv_reg_df(args)    # load fmri data from file

    # correlation_measure = ConnectivityMeasure(kind='correlation')
    # correlation_matrix = correlation_measure.fit_transform([bhv_data[0]['fmri']])[0]

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([dataset.T])[0]

    power = pd.read_csv('coords/Schaefer2018_300Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv')
    coords = np.vstack((power['R'], power['A'], power['S'])).T

    hc_top_k = find_top_k_connections(correlation_matrix, top_50=False, top_100=True)
    fc_top = np.zeros_like(correlation_matrix)
    for i, j in hc_top_k:
        fc_top[i][j] = correlation_matrix[i][j]
    view = plotting.view_connectome(fc_top, coords, edge_threshold='98%', node_size=5)
    view.save_as_html("templates/3d-brain.html")
    return render_template("3d-brain.html")
    


"""
Main handler that returns the architecture of the deep learning model for a particular behaviour
TODO: Generate the architecture diagrams beforehand using online tool: 
"""
@app.route("/architecture/<string:behaviour>")
def show_architecture(behaviour):
    # TODO: replace test images with actual ones later
    if (behaviour == 'ListSort_Unadj'):
        return send_file('images/architecture/working-memory.png', mimetype='image/png')
    elif (behaviour == 'ProcSpeed_Unadj'):
        return send_file('images/architecture/processing-speed.png', mimetype='image/png')
    elif (behaviour == 'PMAT24_A_CR'):
        return send_file('images/architecture/fluid-intelligence.png', mimetype='image/png')
    
    # return send_file('images/architecture/' + behaviour  + '-test.png', mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
    # app.run(host='0.0.0.0', port=5000)

