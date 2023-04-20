#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
import joblib
import numpy 
import numpy as np
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import load_model
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import tensorflow as tf
from keras.layers import LSTM, Masking
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features_BGF = list()
    features_Corr = list()
    features_SEB = list()
    hours_info = list()
    hours_len = list()
    outcomes = list()
    cpcs = list()
    patient_features = list()

    PN = num_patients
    N = 0

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))
        
        # Load data.
        patient_id = patient_ids[i]
        #current_cpc = get_cpc(patient_metadata)
        patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id)

        # Extract features.
        current_features,current_features_BGF, current_features_Corr, current_features_SEB, hour_info, curent_patient_features = get_features(patient_metadata, recording_metadata, recording_data)
        t_len = 0

        for j in range(len(hour_info)):
            if np.all(current_features[j] == 0):
               k = 0
            else:
               t_len = t_len + 1
               features_BGF.append(current_features_BGF[j])
               features_Corr.append(current_features_Corr[j])
               features_SEB.append(current_features_SEB[j])
               hours_info.append(hour_info[j])
               # Extract labels.
               current_cpc = get_cpc(patient_metadata)
               current_outcome = get_outcome(patient_metadata)
               outcomes.append([current_outcome, current_cpc])
               cpcs.append(current_cpc)
               N = N + 1

        hours_len.append(t_len)
        patient_features.append(curent_patient_features)
    
    features_BGF = np.vstack(features_BGF)
    features_Corr = np.vstack(features_Corr)
    features_SEB = np.vstack(features_SEB)
    hours_info = np.vstack(hours_info)
    patient_features =  np.vstack(patient_features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    seq_len = 30000
    seq_wid = 18
    # Impute any missing features; use the mean value by default.
    imputer_BGF = SimpleImputer().fit(features_BGF)
    imputer_Corr = SimpleImputer().fit(features_Corr)
    imputer_SEB = SimpleImputer().fit(features_SEB)
    # Train the models.
    features_BGF = np.array(imputer_BGF.transform(features_BGF))
    features_Corr = np.array(imputer_Corr.transform(features_Corr))
    features_SEB = np.array(imputer_SEB.transform(features_SEB))
    cpcs = cpcs.reshape(len(cpcs),1)
    
    def CNN_models(features_BGF, features_Corr, features_SEB, hours_info, outcomes,N_i,N_f):
        ############################################################################################################################
        ########################  CNN training on the background features ##########################################################
        X = list()

        for i in range(N_i,N_f,1):
            X.append(features_BGF[i*85:(i+1)*85][:])

        X = array(X)

        n_steps = 85
        n_features = 18

        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, outcomes, epochs=500, verbose=1)

        CNN_model_BGF = model
        outcomes_hat_BGF = model.predict(X)
        #########################################################################################################################
        ############################################ CNN training of Pearson's coefficent as input ##############################
        X = list()

        for i in range(N_i,N_f,1):
            X.append(features_Corr[i*18:(i+1)*18][:])

        X = array(X)

        n_steps = 18
        n_features = 18

        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, outcomes, epochs=500, verbose=1)

        CNN_model_Corr = model
        outcomes_hat_Corr = model.predict(X)
        #########################################################################################################################
        ################################### CNN for suppression burst as input ################################################
        X = list()

        for i in range(N_i,N_f,1):
            X.append(features_SEB[i*18:(i+1)*18][:])

        X = array(X)
        
        n_steps = 18
        n_features = 30

        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, outcomes, epochs=500, verbose=1)

        CNN_model_SEB = model
        outcomes_hat_SEB = model.predict(X)
        return outcomes_hat_BGF, outcomes_hat_Corr, outcomes_hat_SEB,CNN_model_BGF, CNN_model_Corr, CNN_model_SEB

    def LSTM_models(patient_features,outcomes_hat_BGF,outcomes_hat_Corr,outcomes_hat_SEB,hours_info,outcomes,PN_i,PN_f,PN):
    ############################################################################################################################
    ############################## Prepare data to feed TS CNN output to LSTM ################################################
        X = list()
        y = list()

        X1 = list()
        y1 = list()

        m = 0

        for i in range(0,PN,1):
            seq = list()
            for j in range(72):
                if j<(hours_len[i]):
                   if i>=PN_i and i<PN_f:
                      dum = 0
                   m = m + 1
                else:
                   if i>=PN_i and i<PN_f:
                      seq.append([-10,-10,-10])
            if i>=PN_i and i<PN_f:
               X.append(seq)
               y.append([outcomes[m-1][0],outcomes[m-1][1]])
            n = 0
            seq = list()
            for j in range(8):
                if i>=PN_i and i<PN_f:
                   if ~np.isnan(patient_features[i][j]):
                       seq.append([patient_features[i][j],j])
                       n = n + 1
            for j in range(n,8,1):
                if i>=PN_i and i<PN_f:
                    seq.append([-10,-10])
            if i>=PN_i and i<PN_f:
               X1.append(seq)
               y1.append([outcomes[m-1][0],outcomes[m-1][1]])

        X1 = array(X1)
        y1 = array(y1)
        ############################################################################################################################
        #######################################  Feed Patient features CNN output to LSTM  #########################################
        special_value = -10.0
        max_seq_len = 8
        dimension = 2
        lstm_units = 128

        model = Sequential()
        model.add(Masking(mask_value=special_value, input_shape=(max_seq_len, dimension)))
        model.add(LSTM(lstm_units))
        model.add(Dense(2, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X1, y1, epochs=1000, verbose=1)

        yhat_LS_PF = model.predict(X1)
        LSTM_PF = model
        ############################################################################################################################
        ###################################### Feed BGF CNN output to LSTM  ##################################################
        X = list()
        y = list()

        m = 0

        for i in range(0,PN,1):
            seq = list()
            for j in range(72):
                if j<(hours_len[i]):
                   if i>=PN_i and i<PN_f:
                      seq.append([outcomes_hat_BGF[m][0],outcomes_hat_BGF[m][1],hours_info[m][0]])
                   m = m + 1
                else:
                   if i>=PN_i and i<PN_f:
                      seq.append([-10,-10,-10])
            if i>=PN_i and i<PN_f:
               X.append(seq)
               y.append([outcomes[m-1][0],outcomes[m-1][1]])
    
        X = array(X)
        y = array(y)

        special_value = -10.0
        max_seq_len = 72
        dimension = 3
        lstm_units = 128

        model = Sequential()
        model.add(Masking(mask_value=special_value, input_shape=(max_seq_len, dimension)))
        model.add(LSTM(lstm_units))
        model.add(Dense(2, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=1000, verbose=1)
           
        yhat_BGF = model.predict(X)
        LSTM_BGF = model
        ##########################################################################################################################
        ###############################################  Feed Pearson CNN ouput into LSTM ########################################
        X = list()
        y = list()
        m = 0

        for i in range(0,PN,1):
            seq = list()
            for j in range(72):
                if j<(hours_len[i]):
                   if i>=PN_i and i<PN_f:
                      seq.append([outcomes_hat_Corr[m][0],outcomes_hat_Corr[m][1],hours_info[m][0]])
                   m = m + 1
                else:
                   if i>=PN_i and i<PN_f:
                      seq.append([-10,-10,-10])
            if i>=PN_i and i<PN_f:
               X.append(seq)
               y.append([outcomes[m-1][0],outcomes[m-1][1]])


        X = array(X)
        y = array(y)

        special_value = -10.0
        max_seq_len = 72
        dimension = 3
        lstm_units = 128

        model = Sequential()
        model.add(Masking(mask_value=special_value, input_shape=(max_seq_len, dimension)))
        model.add(LSTM(lstm_units))
        model.add(Dense(2, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=1000, verbose=1)
        
        yhat_Corr = model.predict(X)
        LSTM_Corr = model
        ##############################################################################################################################
        ################################### CNN into LSTM for SEB ###########################################################
        X = list()
        y = list()

        m = 0

        for i in range(0,PN,1):
            seq = list()
            for j in range(72):
                if j<(hours_len[i]):
                   if i>=PN_i and i<PN_f:
                      seq.append([outcomes_hat_SEB[m][0],outcomes_hat_SEB[m][1],hours_info[m][0]])
                   m = m + 1
                else:
                   if i>=PN_i and i<PN_f:
                      seq.append([-10,-10,-10])
            if i>=PN_i and i<PN_f:
               X.append(seq)
               y.append([outcomes[m-1][0],outcomes[m-1][1]])


        X = array(X)
        y = array(y)

        special_value = -10.0
        max_seq_len = 72
        dimension = 3
        lstm_units = 128

        model = Sequential()
        model.add(Masking(mask_value=special_value, input_shape=(max_seq_len, dimension)))
        model.add(LSTM(lstm_units))
        model.add(Dense(2, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=1000, verbose=1)

        yhat_SEB = model.predict(X)
        LSTM_SEB = model
        return yhat_LS_PF, yhat_BGF, yhat_Corr, yhat_SEB, LSTM_PF, LSTM_BGF, LSTM_Corr, LSTM_SEB
        
    ###############################################################################################################    
    #################################################################################################################################
    N_i,N_f =  0,N
    outcomes_hat_BGF, outcomes_hat_Corr, outcomes_hat_SEB, CNN_model_BGF, CNN_model_Corr, CNN_model_SEB = CNN_models(features_BGF, features_Corr, features_SEB, hours_info, outcomes,N_i,N_f)
    ################################################################################################################################
    ##################################################################################################################################
    
    PN_i,PN_f = 0,PN
    yhat_LS_PF, yhat_BGF, yhat_Corr, yhat_SEB, LSTM_PF, LSTM_BGF, LSTM_Corr, LSTM_SEB = LSTM_models( patient_features, outcomes_hat_BGF, outcomes_hat_Corr, outcomes_hat_SEB, hours_info, outcomes,PN_i,PN_f,PN)
    ###############################################################################################################################
    ###############################################################################################################################
    save_challenge_model(model_folder, imputer_BGF, imputer_Corr, imputer_SEB, CNN_model_BGF, CNN_model_Corr, CNN_model_SEB, LSTM_PF, LSTM_BGF, LSTM_Corr, LSTM_SEB )
    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    # Load models
    models_filename = os.path.join(model_folder, 'imp_models.sav')
    models = joblib.load(models_filename)
    # Load CNN models

    cnn_bgf_filename = os.path.join(model_folder, 'CNN_BGF_model.h5')
    models['CNN_BGF'] = load_model(cnn_bgf_filename)

    cnn_corr_filename = os.path.join(model_folder, 'CNN_Corr_model.h5')
    models['CNN_Corr'] = load_model(cnn_corr_filename)

    cnn_seb_filename = os.path.join(model_folder, 'CNN_SEB_model.h5')
    models['CNN_SEB'] = load_model(cnn_seb_filename)

    # Load LSTM models
    lstm_pf_filename = os.path.join(model_folder, 'LSTM_PF_model.h5')
    models['LSTM_PF'] = load_model(lstm_pf_filename)

    lstm_bgf_filename = os.path.join(model_folder, 'LSTM_BGF_model.h5')
    models['LSTM_BGF'] = load_model(lstm_bgf_filename)

    lstm_corr_filename = os.path.join(model_folder, 'LSTM_Corr_model.h5')
    models['LSTM_Corr'] = load_model(lstm_corr_filename)

    lstm_seb_filename = os.path.join(model_folder, 'LSTM_SEB_model.h5')
    models['LSTM_SEB'] = load_model(lstm_seb_filename)
 
    return models        
  
# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer_BGF = models['imputer_BGF']
    imputer_Corr = models['imputer_Corr']
    imputer_SEB = models['imputer_SEB']
    CNN_BGF = models['CNN_BGF']
    CNN_Corr = models['CNN_Corr']
    CNN_SEB = models['CNN_SEB']
    LSTM_PF = models['LSTM_PF'] 
    LSTM_BGF = models['LSTM_BGF']
    LSTM_Corr = models['LSTM_Corr']
    LSTM_SEB = models['LSTM_SEB']
    # Load data.
    patient_metadata, recording_metadata, recording_data = load_challenge_data(data_folder, patient_id) 

    features_BGF = list()
    features_Corr = list()
    features_SEB = list()
    hours_info = list()
    hours_len = list()
    outcomes = list()
    cpcs = list()
    patient_features = list()

    # Extract features.
    current_features, current_features_BGF, current_features_Corr, current_features_SEB, hour_info, curent_patient_features = get_features(patient_metadata, recording_metadata, recording_data)
    t_len = 0
    
    t_len = 0
    
    k = 0
    for j in range(len(hour_info)):
        if np.all(current_features[j] == 0):#and i<200:
           k = k + 1
        else:
           t_len = t_len + 1
           features_BGF.append(current_features_BGF[j])
           features_Corr.append(current_features_Corr[j])
           features_SEB.append(current_features_SEB[j])
           hours_info.append(hour_info[j])
           # Extract labels.
       
    swi = 1
    if len(hour_info)== k:
       swi = 0

    patient_features.append(curent_patient_features)
    patient_features =  np.vstack(patient_features)
    hours_len.append(t_len) 
    hours_len = np.vstack(hours_len)

    if len(hour_info)!=0 and swi==1: 
       features_BGF = np.vstack(features_BGF)
       features_Corr = np.vstack(features_Corr)
       features_SEB = np.vstack(features_SEB)
       hours_info = np.vstack(hours_info)
       features_BGF = np.array(imputer_BGF.transform(features_BGF))
       features_Corr = np.array(imputer_Corr.transform(features_Corr))
       features_SEB = np.array(imputer_SEB.transform(features_SEB))

    #######################################################################
    #########################################################################
       X = list()

       for i in range(len(hours_info)):
           X.append(features_BGF[i*85:(i+1)*85][:])

       X = array(X)
       outcomes_hat_BGF = CNN_BGF.predict(X)
    ###################################################################################
       X = list()

       for i in range(len(hours_info)):
           X.append(features_Corr[i*18:(i+1)*18][:])

       X = array(X) 
       outcomes_hat_Corr = CNN_Corr.predict(X)
    ##############################################################################
       X = list()

       for i in range(len(hours_info)):
           X.append(features_SEB[i*18:(i+1)*18][:])

       X = array(X)     
       outcomes_hat_SEB = CNN_SEB.predict(X)  
    
    #######################################################################################################################
    ############################## Prepare data to feed TS CNN output to LSTM ################################################
    X = list()
    X1 = list()

    m = 0

    for i in range(0,1,1):
        seq = list()
        for j in range(72):
            if j<(hours_len[i]):
               if i>=0 and i<1:
                  dum = 0
               m = m + 1
            else:
               if i>=0 and i<1:
                  seq.append([-10,-10,-10])
        if i>=0 and i<1:
           X.append(seq)
        n = 0
        seq = list()
        for j in range(8):
            if i>=0 and i<1:
               if ~np.isnan(patient_features[i][j]):
                   seq.append([patient_features[i][j],j])
                   n = n + 1
        for j in range(n,8,1):
            if i>=0 and i<1:
               seq.append([-10,-10])
        if i>=0 and i<1:
           X1.append(seq)

    ##########################################################################################################################
    X1 = array(X1)   
    yhat_LS_PF = LSTM_PF.predict(X1)

    ############################## Prepare the input for LSTM ##################################################################
    X = list()

    m = 0

    for i in range(0,1,1):
        seq = list()
        for j in range(72):
            if j<(hours_len[i]):
               if i>=0 and i<1:
                  seq.append([outcomes_hat_BGF[m][0],outcomes_hat_BGF[m][1],hours_info[m][0]])
               m = m + 1
            else:
               if i>=0 and i<1:
                  seq.append([-10,-10,-10])
        if i>=0 and i<1:
           X.append(seq)

    X = array(X)
    yhat_LS_BGF = LSTM_BGF.predict(X)  
    #####################################################################################################################
    X = list()

    m = 0

    for i in range(0,1,1):
        seq = list()
        for j in range(72):
            if j<(hours_len[i]):
               if i>=0 and i<1:
                  seq.append([outcomes_hat_Corr[m][0],outcomes_hat_Corr[m][1],hours_info[m][0]])
               m = m + 1
            else:
               if i>=0 and i<1:
                  seq.append([-10,-10,-10])
        if i>=0 and i<1:
           X.append(seq)
           
    
    X = array(X)

    yhat_LS_Corr = LSTM_Corr.predict(X)      
    ##########################################################################################################################
        
    X = list()

    m = 0

    for i in range(0,1,1):
        seq = list()
        for j in range(72):
            if j<(hours_len[i]):
               if i>=0 and i<1:
                  seq.append([outcomes_hat_SEB[m][0],outcomes_hat_SEB[m][1],hours_info[m][0]])
               m = m + 1
            else:
               if i>=0 and i<1:
                  seq.append([-10,-10,-10])
        if i>=0 and i<1:
           X.append(seq)
           
    
    X = array(X)
    
    yhat_LS_SEB = LSTM_SEB.predict(X)      
    ###################################################################################################################
    outcome_probability = np.clip((yhat_LS_PF[i][0]*1  +  yhat_LS_BGF[i][0] + yhat_LS_Corr[i][0]*1+ yhat_LS_SEB[i][0]*1)/4.0,0,1)
    
    for i in range(1):
        if yhat_LS_PF[i][0]<=0.5:
           yhat_LS_PF[i][0]=0
        else:
           yhat_LS_PF[i][0]=1
        if yhat_LS_BGF[i][0]<=0.5:
           yhat_LS_BGF[i][0] = 0
        else:
           yhat_LS_BGF[i][0] = 1
        if yhat_LS_Corr[i][0]<=0.5:
           yhat_LS_Corr[i][0] = 0
        else:
           yhat_LS_Corr[i][0] = 1
        if yhat_LS_SEB[i][0] <=0.5:
           yhat_LS_SEB[i][0] = 0
        else:
           yhat_LS_SEB[i][0] = 1
        if (outcome_probability) >=0.5:
           outcome = 1
        else:
           outcome = 0
        cpc = (yhat_LS_PF[i][1]*1 +  yhat_LS_BGF[i][1] + yhat_LS_Corr[i][1]*1+ yhat_LS_SEB[i][1]*1)/4

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc
################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer_BGF, imputer_Corr, imputer_SEB, CNN_BGF, CNN_Corr, CNN_SEB, LSTM_PF, LSTM_BGF, LSTM_Corr, LSTM_SEB):
     # Save imputer models
    imp = {'imputer_BGF': imputer_BGF, 'imputer_Corr': imputer_Corr, 'imputer_SEB': imputer_SEB}
    imp_filename = os.path.join(model_folder, 'imp_models.sav')
    joblib.dump(imp, imp_filename, protocol=0)
    
    # Save CNN models
    cnn_bgf_filename = os.path.join(model_folder, 'CNN_BGF_model.h5')
    CNN_BGF.save(cnn_bgf_filename)

    cnn_corr_filename = os.path.join(model_folder, 'CNN_Corr_model.h5')
    CNN_Corr.save(cnn_corr_filename)

    cnn_seb_filename = os.path.join(model_folder, 'CNN_SEB_model.h5')
    CNN_SEB.save(cnn_seb_filename)

    # Save LSTM models
    lstm_pf_filename = os.path.join(model_folder, 'LSTM_PF_model.h5')
    LSTM_PF.save(lstm_pf_filename)

    lstm_bgf_filename = os.path.join(model_folder, 'LSTM_BGF_model.h5')
    LSTM_BGF.save(lstm_bgf_filename)

    lstm_corr_filename = os.path.join(model_folder, 'LSTM_Corr_model.h5')
    LSTM_Corr.save(lstm_corr_filename)

    lstm_seb_filename = os.path.join(model_folder, 'LSTM_SEB_model.h5')
    LSTM_SEB.save(lstm_seb_filename)
       


# Extract features from the data.
def get_features(patient_metadata, recording_metadata, recording_data):
    # Extract features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    # Combine the patient features.
    patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])

    # Extract features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)

    # Compute mean and standard deviation for each channel for each recording.
    available_signal_data = []
    available_signal_data_BGF = []
    available_signal_data_Corr = []
    available_signal_data_SEB = []
    hours_info = []

    j = 0

    for i in range(num_recordings):
        j = j + 1
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.
            signal_data = signal_data/40.0
            available_signal_data.append(signal_data.transpose()) 
            hours_info.append(j)

            delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
            theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
            alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
            beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

            available_signal_data_BGF.append( np.vstack( (delta_psd.transpose() ,theta_psd.transpose(), alpha_psd.transpose(), beta_psd.transpose() ) ) )
            available_signal_data_Corr.append( np.corrcoef(signal_data.transpose(), rowvar=False) )
            ############################## Burst in EEG ##########################################################
            signal_data1 = array( signal_data.transpose() )
            signal_data1 = signal_data1 * 40
            
            SEB_d = []

            for k in range(18):
                eeg_data = signal_data1[:,k]
                # quantify the amount of suppression bursts in the EEG data
                epoch_suppression = quantify_suppression_bursts(eeg_data)
                SEB_d.append(epoch_suppression)
            
            SEB_d = array(SEB_d)
            mean = np.mean(SEB_d, axis=0)
            std = np.std(SEB_d, axis=0)

            # Column-wise mean center the matrix
            centered_matrix = SEB_d - mean

            # Normalize the centered matrix
            SEB_d = centered_matrix / std
            SEB_d = np.nan_to_num(SEB_d, nan=0)

            available_signal_data_SEB.append(SEB_d)
    return available_signal_data,available_signal_data_BGF, available_signal_data_Corr, available_signal_data_SEB, hours_info, patient_features

def quantify_suppression_bursts(eeg_data):
    """
    Calculates the amount of suppression bursts in the EEG data.

    Parameters:
    - eeg_data: a numpy array containing the EEG data
    - threshold: the threshold value for detecting suppression bursts (default: 50 uV)
    - epoch_duration: the duration of each epoch in seconds (default: 2)

    Returns:
    - a numpy array containing the percentage of time that the EEG signal is below the threshold value for each epoch
    """
    threshold=50
    epoch_duration= 10
    sampling_rate = 100 # assume a sampling rate of 250 Hz
    epoch_samples = epoch_duration * sampling_rate

    # divide the EEG data into non-overlapping epochs
    num_epochs = len(eeg_data) // epoch_samples
    #print( " The number of epochs is " + str(eeg_data) )
    eeg_epochs = np.array_split(eeg_data[:num_epochs*epoch_samples], num_epochs)

    # calculate the percentage of time that the EEG signal is below the threshold value for each epoch
    epoch_suppression = []
    for epoch in eeg_epochs:
        percent_suppression = np.sum(np.abs(epoch) < np.abs(threshold)) / epoch_samples * 1
        epoch_suppression.append(percent_suppression)

    return np.array(epoch_suppression)

#!/usr/bin/env python
