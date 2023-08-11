#Import stuff
import uproot4
import numpy as np
import awkward as ak
from scipy.stats import norm
from scipy.optimize import curve_fit
import os
import copy

import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential, load_model

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split

import optparse
import importlib
import pathlib
from keras import optimizers
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv1D
from qkeras.utils import load_qmodel

import hist
from hist import Hist

import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import cm
import mplhep as hep
plt.style.use(hep.style.ROOT)

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

#line thickness
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 5

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
    except RuntimeError as e:
        print(e)

import random

print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))

def create_train_test_data(dir_path, test_index=400000, train = True):
        
    #Might have to change the version for other ntuple files
    sig = uproot4.open(dir_path+"/test_sig_v12_emseed.root")
    bkg = uproot4.open(dir_path+"/test_bkg_v12_emseed.root")
    qcd = uproot4.open(dir_path+"/test_qcd_v12_emseed.root")
    
    if train:
        sig_input = sig['ntuplePupSingle']['tree']['m_inputs'].array()[:test_index]
        bkg_input = bkg['ntuplePupSingle']['tree']['m_inputs'].array()[:test_index]
        qcd_input = qcd['ntuplePupSingle']['tree']['m_inputs'].array()

        truth_pt_sig = np.asarray(sig['ntuplePupSingle']['tree']['genpt1'].array()[:test_index])
        truth_pt_bkg = np.asarray(bkg['ntuplePupSingle']['tree']['genpt1'].array()[:test_index])
        truth_pt_qcd = np.asarray(qcd['ntuplePupSingle']['tree']['genpt1'].array())

        reco_pt_sig = sig['ntuplePupSingle']['tree']['pt'].array()[:test_index]
        deltaR_sig = sig['ntuplePupSingle']['tree']['gendr1'].array()[:test_index]
        eta_sig = sig['ntuplePupSingle']['tree']['geneta1'].array()[:test_index]
        selection_sig = (reco_pt_sig > 0.) & (abs(deltaR_sig) < 0.4) & (abs(eta_sig) < 2.4)
        y_sig_pT = truth_pt_sig[selection_sig]

        reco_pt_bkg = bkg['ntuplePupSingle']['tree']['pt'].array()[:test_index]
    else:
        sig_input = sig['ntuplePupSingle']['tree']['m_inputs'].array()[test_index:]
        bkg_input = bkg['ntuplePupSingle']['tree']['m_inputs'].array()[test_index:]
        qcd_input = qcd['ntuplePupSingle']['tree']['m_inputs'].array()

        truth_pt_sig = np.asarray(sig['ntuplePupSingle']['tree']['genpt1'].array()[test_index:])
        truth_pt_bkg = np.asarray(bkg['ntuplePupSingle']['tree']['genpt1'].array()[test_index:])
        truth_pt_qcd = np.asarray(qcd['ntuplePupSingle']['tree']['genpt1'].array())
        reco_pt_sig = sig['ntuplePupSingle']['tree']['pt'].array()[test_index:]
        deltaR_sig = sig['ntuplePupSingle']['tree']['gendr1'].array()[test_index:]
        eta_sig = sig['ntuplePupSingle']['tree']['geneta1'].array()[test_index:]
        selection_sig = (reco_pt_sig > 0.) & (abs(deltaR_sig) < 0.4) & (abs(eta_sig) < 2.4)
        y_sig_pT = truth_pt_sig[selection_sig]

        reco_pt_bkg = bkg['ntuplePupSingle']['tree']['pt'].array()[test_index:]
        
    
    selection_bkg = reco_pt_bkg > 10
    y_bkg_pT = truth_pt_bkg[selection_bkg]
    reco_pt_qcd = qcd['ntuplePupSingle']['tree']['pt'].array()
    selection_qcd = reco_pt_qcd > 10
    y_qcd_pT = truth_pt_qcd[selection_qcd]
        
    #Inputs: pt, eta, phi, particle id(one hot encoded)
    X_sig = np.nan_to_num(np.asarray(sig_input[selection_sig]))
    y_sig = np.full(X_sig.shape[0], 1.)
    sig_pt = np.asarray(reco_pt_sig[selection_sig])
    
    X_bkg = np.nan_to_num(np.asarray(bkg_input)[selection_bkg])
    y_bkg = np.full(X_bkg.shape[0], 0.)
    bkg_pt = np.asarray(reco_pt_bkg[selection_bkg])
    
    X_qcd = np.nan_to_num(np.asarray(qcd_input)[selection_qcd])
    y_qcd = np.full(X_qcd.shape[0], 0.)
    qcd_pt = np.asarray(reco_pt_qcd[selection_qcd])
    
    background_pt = np.concatenate([bkg_pt, qcd_pt])
    
    print(y_sig, y_bkg, y_qcd)
    
    if train:
        X_bkg = list(X_bkg)
        y_bkg = list(y_bkg)
        y_bkg_pT = list(y_bkg_pT)
        percent = 0.7
        for _ in range(int(percent*len(X_bkg))):
            n = len(X_bkg)
            random_ind = random.randint(0, n - 1)
            del X_bkg[random_ind]
            del y_bkg[random_ind]
            del y_bkg_pT[random_ind]
        X_bkg = np.asarray(X_bkg)
        y_bkg = np.asarray(y_bkg)
        y_bkg_pT = np.asarray(y_bkg_pT)
        
        X_qcd = list(X_qcd)
        y_qcd = list(y_qcd)
        y_qcd_pT = list(y_qcd_pT)
        for _ in range(int(percent*len(X_qcd))):
            n = len(X_qcd)
            random_ind = random.randint(0, n - 1)
            del X_qcd[random_ind]
            del y_qcd[random_ind]
            del y_qcd_pT[random_ind]
        X_qcd = np.asarray(X_qcd)
        y_qcd = np.asarray(y_qcd)
        y_qcd_pT = np.asarray(y_qcd_pT)

    X_train = np.concatenate([X_sig, X_bkg, X_qcd])
    y_train_jetID = np.concatenate([y_sig, y_bkg, y_qcd])
    MinBias_pT_1 = [1 for i in y_bkg_pT]
    qcd_pT_1 = [1 for i in y_qcd_pT]
#     y_train_pT = np.concatenate([y_sig_pT / sig_pt, y_bkg_pT / bkg_pT, y_qcd_pT / qcd_pt])
    y_train_pT = np.concatenate([y_sig_pT / sig_pt, MinBias_pT_1, qcd_pT_1])
    pt_array = np.concatenate([sig_pt, bkg_pt, qcd_pt])
    
    X_train[abs(X_train) > 1e+9] = 0.
    
    assert not np.any(np.isnan(X_train))
    assert not np.any(np.isnan(y_train_jetID))
    assert not np.any(np.isnan(y_train_pT))
    
    return X_train, y_train_jetID, y_train_pT

X_train_jetID, y_train_jetID,y_train_pT_regress = create_train_test_data("../../ntuples/Jan_25_2023", train=True)
X_test_jetID, y_test_jetID, y_test_pT_regress = create_train_test_data("../../ntuples/Jan_25_2023", train=False)
X_train = X_train_jetID
y_train = y_train_pT_regress
X_test = X_test_jetID
y_test = y_test_pT_regress

from keras.layers import *
from qkeras import *


model_id_name = '../../models/Feb_4_2023_JetMetTalk_v1_pTShape_EMSeed.h5'
model_id = load_model(model_id_name)
minbias_path = "../../ntuples/Jan_25_2023/test_bkg_v12_emseed.root"
sig_path = "../../ntuples/Jan_25_2023/test_sig_v12_emseed.root"
TreeName='ntuplePupSingle'

modelID = load_model(model_id_name)

truth_sig_pt_cut=75,
minbias_pt_cut_value=75
test_index=400000

#Load the data
MinBias = uproot4.open(minbias_path)
sig = uproot4.open(sig_path)

#Signal data prep
truth_pt_sig = sig['ntuplePupSingle']['tree']['genpt1'].array()[test_index:]
truth_deltaR_sig = sig['ntuplePupSingle']['tree']['gendr1'].array()[test_index:]
truth_eta_sig = sig['ntuplePupSingle']['tree']['geneta1'].array()[test_index:]

selection_sig = (truth_pt_sig > truth_sig_pt_cut) & (abs(truth_deltaR_sig) < 0.4) &(abs(truth_eta_sig) < 2.4)

sig_id = np.asarray(sig['ntuplePupSingle']['tree']['event'].array()[test_index:][selection_sig])
sig_pt = np.asarray(sig['ntuplePupSingle']['tree']['pt'].array()[test_index:][selection_sig])
sig_input = np.asarray(sig['ntuplePupSingle']['tree']['m_inputs'].array()[test_index:][selection_sig])      

#Pick out unique events
minbias_id, minbias_index = np.unique(np.asarray(MinBias[TreeName]['tree']['event'].array()[test_index:]), return_index=True)
sig_id_temp, sig_index_temp = np.unique(sig_id, return_index=True)

true_sig_id, _, true_signal_index = np.intersect1d(np.unique(minbias_id), np.unique(sig_id_temp), return_indices=True)

#True signal data preparation (these are events that actually overlaps with MinBias)
true_sig_pt = sig_pt[sig_index_temp][true_signal_index]
true_sig_input = sig_input[sig_index_temp][true_signal_index]
true_sig_score = modelID.predict(true_sig_input).flatten()

minbias_pt = MinBias['ntuplePupSingle']['tree']['pt'].array()[test_index:][minbias_index]
minbias_input = np.asarray(MinBias['ntuplePupSingle']['tree']['m_inputs'].array()[test_index:][minbias_index])
minbias_input[abs(minbias_input) > 1e+9] = 0.

minbias_score = modelID.predict(minbias_input).flatten()

#
sig_pt_cut = true_sig_pt > minbias_pt_cut_value
minbias_pt_cut = minbias_pt > minbias_pt_cut_value

true_sig_score_pt = true_sig_score[sig_pt_cut]
minbias_score_pt = minbias_score[minbias_pt_cut]

#Now apply correction
#model_regress = load_model(ModelNameRegress)

#minbias_pt_corrected_ratio = model_regress.predict(minbias_input).flatten()
#minbias_pt_corrected = np.multiply(minbias_pt, minbias_pt_corrected_ratio)

#true_sig_pt_corrected_ratio = model_regress.predict(true_sig_input).flatten()
#true_sig_pt_corrected = np.multiply(true_sig_pt, true_sig_pt_corrected_ratio)

#sig_pt_cut_corrected = true_sig_pt_corrected > minbias_pt_cut_value
#minbias_pt_cut_corrected = minbias_pt_corrected > minbias_pt_cut_value

#true_sig_score_pt_corrected = true_sig_score[sig_pt_cut_corrected]
#minbias_score_pt_corrected = minbias_score[minbias_pt_cut_corrected]

n_event = minbias_index.shape[0]
n_sig_event = true_sig_id.shape[0]

##LOOP to calculate the ROC curves
tau_score_edges = [round(i,2) for i in np.arange(0, 0.8, 0.01).tolist()]+\
              [round(i,4) for i in np.arange(0.9, 1, 0.0001)] + [1]

sig_list = []
bkg_list = []

for tau_score_cut in tau_score_edges:

    bkg_pass = sum(minbias_score_pt>tau_score_cut)
    sig_pass = sum(true_sig_score_pt>tau_score_cut)

    sig_list.append(sig_pass/n_sig_event)
    bkg_list.append(bkg_pass/n_event)


bkg_list_scaled = [i*(32e+3) for i in bkg_list]



def prep_rate_data(NormalModelID, 
                   ModelNameID,
             ModelNameRegress,
             truth_sig_pt_cut=50,
             minbias_pt_cut_value=50,
             minbias_path = "../../ntuples/Jan_25_2023/test_bkg_v12_emseed.root",
             sig_path = "../../ntuples/Jan_25_2023/test_sig_v12_emseed.root",
             TreeName='ntuplePupSingle',
             test_index=400000):
    
    
    plt.plot(sig_list, bkg_list_scaled, label=r'Tau NN ID (No $p_T$ Correction)',linewidth=5)

    
    #Load the data
    MinBias = uproot4.open(minbias_path)
    sig = uproot4.open(sig_path)
    
    #Signal data prep
    truth_pt_sig = sig['ntuplePupSingle']['tree']['genpt1'].array()[test_index:]
    truth_deltaR_sig = sig['ntuplePupSingle']['tree']['gendr1'].array()[test_index:]
    truth_eta_sig = sig['ntuplePupSingle']['tree']['geneta1'].array()[test_index:]

    selection_sig = (truth_pt_sig > truth_sig_pt_cut) & (abs(truth_deltaR_sig) < 0.4) &(abs(truth_eta_sig) < 2.4)

    sig_id = np.asarray(sig['ntuplePupSingle']['tree']['event'].array()[test_index:][selection_sig])
    sig_pt = np.asarray(sig['ntuplePupSingle']['tree']['pt'].array()[test_index:][selection_sig])
    sig_input = np.asarray(sig['ntuplePupSingle']['tree']['m_inputs'].array()[test_index:][selection_sig])      
    
    #Pick out unique events
    minbias_id, minbias_index = np.unique(np.asarray(MinBias[TreeName]['tree']['event'].array()[test_index:]), return_index=True)
    sig_id_temp, sig_index_temp = np.unique(sig_id, return_index=True)
    
    true_sig_id, _, true_signal_index = np.intersect1d(np.unique(minbias_id), np.unique(sig_id_temp), return_indices=True)
    
    #True signal data preparation (these are events that actually overlaps with MinBias)
    true_sig_pt = sig_pt[sig_index_temp][true_signal_index]
    true_sig_input = sig_input[sig_index_temp][true_signal_index]
    for i in range(len(ModelNameID)):
        modelID = ModelNameID[i]
        true_sig_score = modelID.predict(true_sig_input)[0].flatten()

        minbias_pt = MinBias['ntuplePupSingle']['tree']['pt'].array()[test_index:][minbias_index]
        minbias_input = np.asarray(MinBias['ntuplePupSingle']['tree']['m_inputs'].array()[test_index:][minbias_index])
        minbias_input[abs(minbias_input) > 1e+9] = 0.

        minbias_score = modelID.predict(minbias_input)[0].flatten()

        #
        sig_pt_cut = true_sig_pt > minbias_pt_cut_value
        minbias_pt_cut = minbias_pt > minbias_pt_cut_value

        true_sig_score_pt = true_sig_score[sig_pt_cut]
        minbias_score_pt = minbias_score[minbias_pt_cut]

        #Now apply correction
    #     for i in range(len(ModelNameRegress)):
        #model_regress = load_model(ModelNameRegress)
        model_regress = ModelNameRegress[i]

        minbias_pt_corrected_ratio = model_regress.predict(minbias_input)[1].flatten()
        minbias_pt_corrected = np.multiply(minbias_pt, minbias_pt_corrected_ratio)

        true_sig_pt_corrected_ratio = model_regress.predict(true_sig_input)[1].flatten()
        true_sig_pt_corrected = np.multiply(true_sig_pt, true_sig_pt_corrected_ratio)

        sig_pt_cut_corrected = true_sig_pt_corrected > minbias_pt_cut_value
        minbias_pt_cut_corrected = minbias_pt_corrected > minbias_pt_cut_value

        true_sig_score_pt_corrected = true_sig_score[sig_pt_cut_corrected]
        minbias_score_pt_corrected = minbias_score[minbias_pt_cut_corrected]

        n_event = minbias_index.shape[0]
        n_sig_event = true_sig_id.shape[0]

        ##LOOP to calculate the ROC curves
        tau_score_edges = [round(i,2) for i in np.arange(0, 0.8, 0.01).tolist()]+\
                      [round(i,4) for i in np.arange(0.9, 1, 0.0001)] + [1]

        sig_list_corrected = []
        bkg_list_corrected = []

        for tau_score_cut in tau_score_edges:
            bkg_pass = sum(minbias_score_pt_corrected>tau_score_cut)
            sig_pass = sum(true_sig_score_pt_corrected>tau_score_cut)

            sig_list_corrected.append(sig_pass/n_sig_event)
            bkg_list_corrected.append(bkg_pass/n_event)

        bkg_list_scaled_corrected = [i*(32e+3) for i in bkg_list_corrected]

        plt.plot(sig_list_corrected, bkg_list_scaled_corrected, label=r'Tau NN ID, $\gamma=$' + str(round(gammas[i], 1)),linewidth=2)
        hep.cms.text("Phase 2 Simulation")
        hep.cms.lumitext("PU 200 (14 TeV)")

        plt.ylabel(r'$Single \tau_h$ Trigger Rate [kHz]')
        plt.xlabel(r'$\tau_h$ [$p_T^{gen} > % d$ GeV]' %int(truth_sig_pt_cut))

        plt.yscale('log')
        #plt.legend(loc='best',fontsize=20)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    plt.plot([],[], 'none', label=r'(MinBias $p_T^{reco (or~corrected)}$ >  % d GeV)'%int(minbias_pt_cut_value))
    plt.savefig('gridscan/qmerged_' + str(num_filters) + "_" + str(kernel_size) + "_"
                          + str(stride_length) + ".png", bbox_inches = "tight")
    plt.show()

    return n_event, n_sig_event, true_sig_score_pt, minbias_score_pt, true_sig_score_pt_corrected, minbias_score_pt_corrected


#####------------------ MODEL TRAINING ------------------#####
bits = 9
bits_int = 2
alpha_val = 1
bits_conv = 9

def quantize_merge(num_filters, kernel_size, strides):
    model = tf.keras.Sequential()
    
    inputs = tf.keras.layers.Input(shape=(80,1,), name='input')

    
    main_branch = QConv1D(filters=20, kernel_size=4, strides=4, name='Conv1D_1',
                  kernel_quantizer=quantized_bits(
                      bits_conv, bits_int, alpha=alpha_val),
                  bias_quantizer=quantized_bits(
                      bits_conv, bits_int, alpha=alpha_val),
                  # kernel_regularizer=l1(0.0001),
                  kernel_initializer='lecun_uniform')(inputs)
    main_branch = QActivation(activation=quantized_relu(bits), name='relu_1')(main_branch)
    main_branch = QConv1D(filters=num_filters, kernel_size=kernel_size, strides = strides, name='Conv1D_2',
                  kernel_quantizer=quantized_bits(
                      bits_conv, bits_int, alpha=alpha_val),
                  bias_quantizer=quantized_bits(
                      bits_conv, bits_int, alpha=alpha_val),
                  kernel_initializer='lecun_uniform')(main_branch)
    main_branch = QActivation(activation=quantized_relu(bits), name='relu_2')(main_branch)   
    main_branch = tf.keras.layers.Flatten()(main_branch)
    print(main_branch)
    main_branch = QDense(25, name='Dense_1',
                     kernel_quantizer=quantized_bits(
                         bits, bits_int, alpha=alpha_val),
                     bias_quantizer=quantized_bits(
                         bits, bits_int, alpha=alpha_val),
                     kernel_initializer='lecun_uniform')(main_branch)  # , kernel_regularizer=l1(0.0001)))
    main_branch = QActivation(activation=quantized_relu(bits), name='relu_3')(main_branch)
    main_branch = QDense(25, name='Dense_2',
                     kernel_quantizer=quantized_bits(
                         bits, bits_int, alpha=alpha_val),
                     bias_quantizer=quantized_bits(
                         bits, bits_int, alpha=alpha_val),
                     kernel_initializer='lecun_uniform')(main_branch)
    main_branch = QActivation(activation=quantized_relu(bits), name='relu_4')(main_branch)
    main_branch = QDense(15, name='Dense_3',
                     kernel_quantizer=quantized_bits(
                         bits, bits_int, alpha=alpha_val),
                     bias_quantizer=quantized_bits(
                         bits, bits_int, alpha=alpha_val),
                     kernel_initializer='lecun_uniform')(main_branch)  # , kernel_regularizer=l1(0.0001)))
    main_branch = QActivation(activation=quantized_relu(bits), name='relu_5')(main_branch)
    main_branch = QDense(15, name='Dense_4',
                     kernel_quantizer=quantized_bits(
                         bits, bits_int, alpha=alpha_val),
                     bias_quantizer=quantized_bits(
                         bits, bits_int, alpha=alpha_val),
                     kernel_initializer='lecun_uniform')(main_branch)
    main_branch = QActivation(activation=quantized_relu(bits), name='relu_6')(main_branch)
    main_branch = QDense(10, name='Dense_5',
                     kernel_quantizer=quantized_bits(
                         bits, bits_int, alpha=alpha_val),
                     bias_quantizer=quantized_bits(
                         bits, bits_int, alpha=alpha_val),
                     kernel_initializer='lecun_uniform')(main_branch)
    main_branch = QActivation(activation=quantized_relu(bits), name='relu_7')(main_branch)
    jetID_branch = QDense(1, name='Dense_6',
                     kernel_quantizer=quantized_bits(
                         16, 2, alpha=alpha_val),
                     bias_quantizer=quantized_bits(
                         16, 2, alpha=alpha_val),
                     kernel_initializer='lecun_uniform')(main_branch)
    jetID_branch = tf.keras.layers.Activation(activation='sigmoid', name='jetID_output')(jetID_branch)
    
    pT_branch = QDense(1, name='pT_output',
                     kernel_quantizer=quantized_bits(
                         16, 4, alpha=alpha_val),
                     bias_quantizer=quantized_bits(
                         16, 4, alpha=alpha_val),
                     kernel_initializer='lecun_uniform')(main_branch)
    
    model = tf.keras.Model(inputs = inputs, outputs = [jetID_branch, pT_branch])

    return model
    
# Coupling Loss Functions

def compile_model(model, gamma):
    opt = optimizers.Adam()
    model.compile(optimizer=opt,
                  loss={'jetID_output': 'binary_crossentropy', 
                        'pT_output': 'mean_squared_error'},
                  loss_weights={'jetID_output': gamma, 
                                'pT_output': 1 - gamma}, 
                  metrics=['accuracy'])
        
    return model

#models = np.zeros((2, 2, 2, 5))
#gammas = np.linspace(0, 1, 2) # 0,1,10
gammas = [0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88]
for num_filters in range(5, 21):
#num_filters = 5
    for kernel_size in range(1, 11): #1, 10
        for stride_length in range(1, 11):#1, 10
            aux_models = []
            for gamma in range(len(gammas)):
                model = quantize_merge(num_filters, kernel_size, stride_length)
                model = compile_model(model, gammas[gamma])
                model.summary()
                #Train the network
                callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=5)
                X_train = X_train_jetID
                history = model.fit({'input': X_train},
                                            {'jetID_output': y_train_jetID, 'pT_output': y_train_pT_regress},

                                    epochs=20,
                                    batch_size=256,
                                    verbose=1,
                                    validation_split=0.20,
                                    callbacks = [callback])
                aux_models.append(model)
                model.save("gridscan/quantized_merged_gridscan_" + str(num_filters) + "_" + str(kernel_size) + "_"
                          + str(stride_length) + "_" + str(gammas[gamma]) + ".h5")
            n_event, n_sig_event, true_sig_score_pt, minbias_score_pt, true_sig_score_pt_corrected, minbias_score_pt_corrected = prep_rate_data(model_id_name,
                                                                                                                                        aux_models,
                                                                                                                                        aux_models,
                                                                                                                                        truth_sig_pt_cut=75,
                                                                                                                                        minbias_pt_cut_value=75)
