import numpy as np
import rsatoolbox
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import loadmat
from sklearn.manifold import MDS
import json

# This script loads in preprocessed firing rate data and uses it to build and save out RDM files. These files are then
# reimported back into MATLAB and plotted as RDMs for figure 2b using the script figure2bc_plotRDMandMDS.m, which also uses them to construct
# the MDS plots in figure 2c.
# Isabelle Rosenthal 2022

# directory where the output file will be saved
saveName = '../Data/touchExploration_RSA.mat'

plotRDM = 0 # to plot the RDM directly in Python

# load in preprocessed data file
matFile = '../Data/touchExploration_preprocessedSpks_forRDM.mat'
matData = loadmat(matFile)
matData = matData["outStruct"]
condNames = ['FPa',  'TPa', 'FPf',  'TPf','obj',
             'VrFPa','BLa','VrFPf','BLf']

# extract variables from the data file
trialClass = matData["trialClass"][0][0]
classID = matData["classID"][0][0][0]
chList = matData["chList"][0][0][0]
win = matData["win"][0][0][0][0]
binnedSpks = matData["binnedSpks"][0][0]
modLabels = matData["label_mod"][0][0]
modLabels = np.array([a[0] for a in modLabels])
condList = np.unique(modLabels)
numConds = np.size(condList)
setLabels = matData["setLabels"][0][0]# marks which 7-condition set each trial comes from (total 7 sets)
setLabels = np.array([a[0] for a in setLabels])
trialClassName = matData["trialClassNames"][0][0]
trialClassName = np.array([a[0][0] for a in trialClassName])

# set timing bins to use for RDMs
binWidth = 0.5 # in sec
binStarts = np.arange(0, 120, 10) # every 0.5 sec
binEnds = np.arange(10, 130, 10) # every 0.5 sec
numBins = np.size(binStarts)
nsp = 2
# # to compute the RDM, need the firing rate for each channel averaged across time within each bin
avgSpks = np.zeros([96, np.shape(binnedSpks)[1], numBins])
for bn in range(numBins):
        avgSpks[:, :, bn] = np.nanmean(binnedSpks[:, :,binStarts[bn]:binEnds[bn]], axis=2)
Lbs = ['-4', '-3.5','3','-2.5', '-2','-1.5','-1','-0.5', '0', '0.5', '1', '1.5', '2']

# construct the RDM using cross-validated Mahalanobis distance (LDS) with multivariate noise normalization in every bin
rdmList = []
for bn in range(numBins):
    # format the data as an rsa dataset with 96 channels
    rsaData = rsatoolbox.data.Dataset(np.squeeze(avgSpks[:,:, bn]).T,
                                      channel_descriptors= {'chList':np.squeeze(chList[0])},
                                      obs_descriptors={'modLabels':modLabels, 'modNames': trialClassName, 'sets':setLabels})

    # multivariate noise normalization
    noise_shrink = rsatoolbox.data.noise.prec_from_measurements(rsaData, obs_desc='modLabels', method='shrinkage_eye')

    # create rdm (representational dissimilarity matrix) with cross-validated Mahalanobis distance
    rdm = rsatoolbox.rdm.calc_rdm(rsaData, method='crossnobis', descriptor='modLabels',  #,noise=noise_shrink)
                                  cv_descriptor='sets',noise=noise_shrink)
    outRDM = rdm.get_matrices()[0]

    # reorder by touch type
    tySort = np.array([0, 2, 6, 8, 5, 7, 1, 3, 4])
    condNames_byType = np.array(condNames)[tySort]
    rdm_byType = np.zeros([numConds,numConds])
    for cc in range(numConds):
        for cr in range(numConds):
            rdm_byType[cc,cr] = outRDM[tySort[cc], tySort[cr]]
    if plotRDM==1:
        # now plot my own rdm
        fig, ax = plt.subplots()
        im = ax.imshow(rdm_byType) #, vmin = 0.0, vmax = 0.05)
        ax.set_xticks(np.arange(len(condNames_byType)))
        ax.set_yticks(np.arange(len(condNames_byType)))
        ax.set_xticklabels(condNames_byType)
        ax.set_yticklabels(condNames_byType)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_title("nsp " + str(nsp+1) + ", CV noise-corrected Mahalanobis RDM:" + Lbs[bn] + " to " + Lbs[bn+1])
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('distances', rotation=-90, va="bottom")
        fig.tight_layout()
        plt.show()
    rdmList.append(outRDM)

outDict = {}
outDict["RDMs"] = rdmList # needs to be sorted on the other end
outDict["timeBins"] = Lbs
scipy.io.savemat(saveName, outDict)