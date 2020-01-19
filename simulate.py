import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import pylab as pl
from scipy.stats.stats import pearsonr


def gen_profiles(num_profiles, genes, n_markers, noise, std=0.05, m_amp=3):
    profiles = []

    for p in range(num_profiles):
        profile = np.random.rand(genes)
        markers = np.random.choice(genes, n_markers, replace=False)
        # increase expression of marker genes
        profile[markers] += np.average(profile)
        profile[markers] *= m_amp
        # two types of noise
        if noise == 'norm':
            n = np.random.normal(0, std, genes)
        else:
            n = np.random.standard_cauchy(genes)
        # add noise to the generated profile
        profile *= 1+n
        # append to profiles
        profiles.append(profile)

    profiles = np.vstack(profiles)
    return profiles.T


def gen_samples(profiles, n_samples, std):
    genes, celltypes = np.shape(profiles)
    # generate proportions for each sample
    proportions = np.random.rand(celltypes, n_samples)
    proportions /= np.sum(proportions, axis=0)
    # create expressions
    samples =  np.dot(profiles, proportions)
    # add measurement noise
    samples += np.random.normal(0, std, (genes, n_samples))

    return samples, proportions.T


#stds = [0.01, 0.1, 0.2, 0.3, 0.5, 1, 2]
#stds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
stds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
num_datasets = 20
num_celltypes = 3
num_markers = 200
num_samples = 500
num_genes = 10000
noise = 'norm' # or cauchy

pca_mse_all = []
ica_mse_all = []
nmf_mse_all = []
rnd_mse_all = []

pca_mse_plt_all = []
ica_mse_plt_all = []
nmf_mse_plt_all = []
rnd_mse_plt_all = []

props_all = []

for std in stds:
    print('std: ' + str(std))
    pca_mse = 0
    ica_mse = 0
    nmf_mse = 0
    rnd_mse = 0

    
    pca_mse_plt = []
    ica_mse_plt = []
    nmf_mse_plt = []
    rnd_mse_plt = []
    
    props = []
    for i in range(num_datasets):
        print('dataset: ' + str(i+1))
        profiles = gen_profiles(num_celltypes, num_genes, num_markers, noise, std)
        samples, proportions = gen_samples(profiles, num_samples, std)
        
        # FEATURE SELECTION USING CONCRETE AUTOENCODER

        # preprocessing
        samples_normalized = normalize(samples, axis=0).T
        if np.min(samples_normalized) < 0:
            samples_normalized -= np.min(samples_normalized)

        # PCA
        pca = PCA(n_components=num_celltypes)
        X_pca = pca.fit_transform(samples_normalized)
        # normalize result
        n = X_pca - np.expand_dims(np.amin(X_pca, axis=1), axis=1)
        res_p = n / np.expand_dims(np.sum(n, axis=1), axis=1)
        pca_mse += mean_squared_error(proportions, res_p)
        # calculate pearson correlation
        pca_mse_plt.append([mean_squared_error(proportions[:,i],res_p[:,i]) for i in range(num_celltypes)])
        

        # ICA
        ica = FastICA(n_components=num_celltypes)
        X_ica = ica.fit_transform(samples_normalized)
        n = X_ica - np.expand_dims(np.amin(X_ica, axis=1), axis=1)
        res_i = n / np.expand_dims(np.sum(n, axis=1), axis=1)
        ica_mse += mean_squared_error(proportions, res_i)
        ica_mse_plt.append([mean_squared_error(proportions[:,i],res_i[:,i]) for i in range(num_celltypes)])
        
        # NMF
        model = NMF(n_components=num_celltypes)
        W = model.fit_transform(samples_normalized)
        res_n = W / np.expand_dims(np.sum(W, axis=1), axis=1)
        nmf_mse  += (mean_squared_error(proportions, res_n))
        nmf_mse_plt.append([mean_squared_error(proportions[:,i],res_n[:,i]) for i in range(num_celltypes)])
        
        # Random proportions
        rnd_prop = np.random.rand(num_celltypes,num_samples)
        rnd_prop /= np.sum(rnd_prop, axis = 0)
        rnd_mse += mean_squared_error(proportions, rnd_prop.T)
        rnd_mse_plt.append([mean_squared_error(proportions[:,i],rnd_prop.T[:,i]) for i in range(num_celltypes)])
        
        props.append([res_p, res_i, res_n, rnd_prop.T, proportions])

    # average over all datasets
    pca_mse_all.append(pca_mse/num_datasets)
    ica_mse_all.append(ica_mse/num_datasets)
    nmf_mse_all.append(nmf_mse/num_datasets)
    rnd_mse_all.append(rnd_mse/num_datasets)
    
    pca_mse_plt_all.append(np.matrix(pca_mse_plt))
    ica_mse_plt_all.append(np.matrix(ica_mse_plt))
    nmf_mse_plt_all.append(np.matrix(nmf_mse_plt))
    rnd_mse_plt_all.append(np.matrix(rnd_mse_plt))
    
    props_all.append(props)


pca_mse = []
ica_mse = []
nmf_mse = []
rnd_mse = []
for i in range(len(stds)):
    pca_mse.append(np.array(np.mean(pca_mse_plt_all[i],axis = 0))[0])
    ica_mse.append(np.array(np.mean(ica_mse_plt_all[i],axis = 0))[0])
    nmf_mse.append(np.array(np.mean(nmf_mse_plt_all[i],axis = 0))[0])
    rnd_mse.append(np.array(np.mean(rnd_mse_plt_all[i],axis = 0))[0])
    
pca_mse = np.matrix(pca_mse)
ica_mse = np.matrix(ica_mse)
nmf_mse = np.matrix(nmf_mse)
rnd_mse = np.matrix(rnd_mse)
    
fig, axs = plt.subplots(1,num_celltypes+1)
for i in range(num_celltypes):
    axs[i].plot(stds,pca_mse[:,i], label='PCA')
    axs[i].plot(stds,ica_mse[:,i], label='ICA')
    axs[i].plot(stds,nmf_mse[:,i], label='NMF')
    axs[i].plot(stds,rnd_mse[:,i], label='Random')
    axs[i].set_xlabel("Standard deviation")
    axs[i].set_ylabel("MSE")
    axs[i].set_title("Cell type: " + str(i+1))
    axs[i].set_ylim([0.03,0.17])
axs[num_celltypes].plot(stds, pca_mse_all, label='PCA')
axs[num_celltypes].plot(stds, ica_mse_all, label='ICA')
axs[num_celltypes].plot(stds, nmf_mse_all, label='NMF')
axs[num_celltypes].plot(stds, rnd_mse_all, label = 'Random')
axs[num_celltypes].set_xlabel("Standard deviation")
axs[num_celltypes].set_ylabel("MSE")
axs[num_celltypes].set_title("Overall")
axs[num_celltypes].set_ylim([0.03,0.17])
handles, labels = axs[num_celltypes].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')


colors = ['blue','green','red','orange','purple']
for k in range(4):
    fig, axs = plt.subplots(1,7)
    fig.set_size_inches(16,3.5)
    for i in range(len(stds)):
        axs[i].set_title("Std: " + str(stds[i]))
        axs[i].tick_params(axis='both', which='major', labelsize=8) 
        axs[i].tick_params(axis='both', which='minor', labelsize=8)
        axs[i].plot([0,1],[0,1])
        for j in range(num_datasets):
            for cl in range(num_celltypes):
                axs[i].scatter(props_all[i][j][4][:,0],props_all[i][j][k][:,cl],c=colors[cl],s=1)
                axs[i].scatter(props_all[i][j][4][:,0],props_all[i][j][k][:,cl],c=colors[cl],s=1)
                axs[i].scatter(props_all[i][j][4][:,0],props_all[i][j][k][:,cl],c=colors[cl],s=1)
            
    fig.text(0.5, 0.04, 'Real proportions', ha='center', va='center')
    fig.text(0.10, 0.5, 'Estimated proportions', ha='center', va='center', rotation='vertical')
    
# Benchmark data            
#gs_data = np.matrix(pd.read_excel(r'GSE19830.xlsx', header=None))
#gs_prop = np.matrix(pd.read_excel(r'gs_prop.xlsx', header= None))
#gs_normalized = normalize(gs_data, axis=0).T
#gs_normalized = gs_data.T
#
#pca = PCA(n_components=3)
#X_pca = pca.fit_transform(gs_normalized)
## normalize result
#n = X_pca - np.expand_dims(np.amin(X_pca, axis=1), axis=1)
#res_p = n / np.expand_dims(np.sum(n, axis=1), axis=1)
## calculate pearson correlation
#
#       
#
## ICA
#ica = FastICA(n_components=3)
#X_ica = ica.fit_transform(gs_normalized)
#n = X_ica - np.expand_dims(np.amin(X_ica, axis=1), axis=1)
#res_i = n / np.expand_dims(np.sum(n, axis=1), axis=1)
#
## NMF
#model = NMF(n_components=3)
#W = model.fit_transform(gs_normalized)
#res_n = W / np.expand_dims(np.sum(W, axis=1), axis=1)
#
#fig,axs = plt.subplots(1,3)
#esti_props = [res_p,res_i,res_n]
#for i,ax in enumerate(axs):
#    ax.scatter(np.array(gs_prop[:,0]),esti_props[i][:,0],c = 'blue')
#    ax.scatter(np.array(gs_prop[:,1]),esti_props[i][:,1],c = 'green')
#    ax.scatter(np.array(gs_prop[:,2]),esti_props[i][:,2],c = 'red')
#    ax.plot([0,1],[0,1])



















    