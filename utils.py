import numpy as np
import pandas as pd
import pickle
from sklearn.covariance import GraphicalLassoCV
from nilearn import connectome


K_RUNS = 4 # movie watching RUNs

def _info(s):
    print('---')
    print(s)
    print('---')

def _get_behavioral(subject_list):
    '''
    load behavioral measures for all HCP subjects
    '''
    bhv_path = 'data/unrestricted_behavioral.csv'
    bhv_df = pd.read_csv(bhv_path)
    
    subjects = subject_list.astype(int)

    # bhv_df = bhv_df.loc[bhv_df['Subject'].isin(subject_list)]
    bhv_df = bhv_df.loc[bhv_df['Subject'].isin(subjects)]
    bhv_df = bhv_df.reset_index(drop=True)

    return bhv_df

def _vectorize(Q):
    '''
    Q: symmetric matrix (FC)
    return: unique elements as an array
    ignore diagonal elements
    '''
    # extract lower triangular matrix
    tri = np.tril(Q, -1)

    vec = []
    for ii in range(1, tri.shape[0]):
        for jj in range(ii):
            vec.append(tri[ii, jj])
    
    return np.asarray(vec)

def _getfc(scan):
    '''
    Functional Connectivity matrix
    using Pearson correlation
    
    scan: timeseries of ROI x t
    output: FC of ROI x ROI
    '''
    return np.corrcoef(scan)


def _get_clip_labels():
    '''
    assign all clips within runs a label
    use 0 for testretest
    '''
    # where are the clips within the run?
    timing_file = pd.read_csv('data/videoclip_tr_lookup.csv')

    clips = []
    for run in range(K_RUNS):
        run_name = 'MOVIE%d' %(run+1) #MOVIEx_7T_yz
        timing_df = timing_file[timing_file['run'].str.contains(run_name)]  
        timing_df = timing_df.reset_index(drop=True)

        for jj, row in timing_df.iterrows():
            clips.append(row['clip_name'])
            
    clip_y = {}
    jj = 1
    for clip in clips:
        if 'testretest' in clip:
            clip_y[clip] = 0
        else:
            clip_y[clip] = jj
            jj += 1

    return clip_y


def _get_clip_lengths():
    '''
    return:
    clip_length: dict of lengths of each clip
    '''
    K_RUNS = 4
    # where are the clips within the run?
    timing_file = pd.read_csv('data/videoclip_tr_lookup.csv')

    clip_length = {}
    for run in range(K_RUNS):

        run_name = 'MOVIE%d' %(run+1) #MOVIEx_7T_yz
        timing_df = timing_file[timing_file['run'].str.contains(run_name)]  
        timing_df = timing_df.reset_index(drop=True)

        for jj, clip in timing_df.iterrows():
            start = int(np.floor(clip['start_tr']))
            stop = int(np.ceil(clip['stop_tr']))
            t_length = stop - start
            clip_length[clip['clip_name']] = t_length
            
    return clip_length

def _bhv_reg_df(args):
    '''
    data for  bhv prediction

    '''

    print('loading data...')
    load_path = (args.input_data + '/data_MOVIE_runs_' +
        'roi_%d_net_%d_ts.pkl' %(args.roi, args.net))

    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    print('loading subject list...')
    subject_list = np.sort(list(data.keys()))
    print(subject_list.shape)
    
    print('loading behavioral data for subjects...')
    # get behavioral data for subject_list
    bhv_df = _get_behavioral(subject_list)
    bhv_df = bhv_df[['Subject', args.bhv]]
    '''
    ***normalize bhv scores
    must be explicitly done for pytorch
    '''
    print('normalizing...')
    b = bhv_df[args.bhv].values
    bhv_df[args.bhv] = (b - np.min(b))/(np.max(b) - np.min(b))


    # where are the clips within the run?
    timing_file = pd.read_csv('data/videoclip_tr_lookup.csv')


    '''
    main
    '''
    clip_y = _get_clip_labels()
    
    table = []
    for run in range(K_RUNS):
        
        print('loading run %d/%d' %(run+1, K_RUNS))
        run_name = 'MOVIE%d' %(run+1) #MOVIEx_7T_yz

        # timing file for run
        timing_df = timing_file[
            timing_file['run'].str.contains(run_name)]  
        timing_df = timing_df.reset_index(drop=True)

        for subject in data:
            # get subject data (time x roi x run)
            roi_ts = data[subject][:, :, run]

            for jj, clip in timing_df.iterrows():

                start = int(np.floor(clip['start_tr']))
                stop = int(np.ceil(clip['stop_tr']))
                
                # assign label to clip
                c = clip_y[clip['clip_name']]
                
                s_data = {}
                s_data['Subject'] = subject
                s_data['c'] = c
                s_data['fmri'] = roi_ts[start : stop, :]
                s_data['bhv'] = bhv_df.loc[bhv_df['Subject']==subject.astype(int)].values.squeeze()[1:]
                
                table.append(s_data)
                
    
    return table   # dimension is 3168 = (18 movie x 176 subj) #df, bhv_df

def _extract_fc(data,kind):
    print('---')
    print('Connectivity measure extraction ...')
    print('---')
    fc_measures = []
    labels = []
    sub_ID = []
    clip = []
    for item in data:
        timeseries = item['fmri']
        fc_sub = subject_connectivity(timeseries, kind)
        fc_measures.append(fc_sub)
        labels.append(item['bhv'][0])
        sub_ID.append(item['Subject'].astype(int))
        clip.append(item['c'])
    print('done!')
    return np.stack(fc_measures), np.stack(labels), np.stack(sub_ID)
        
def subject_connectivity(timeseries, kind):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation

    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    if kind == 'lasso':
        # Graph Lasso estimator
        covariance_estimator = GraphicalLassoCV(verbose=1)
        covariance_estimator.fit(timeseries)
        connectivity = covariance_estimator.covariance_
        print('Covariance matrix has shape {0}.'.format(connectivity.shape))

    elif kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

    return connectivity