import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import pylab as pl


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
        profile += n
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


stds = [0.01, 0.1, 0.2, 0.3, 0.5, 1, 2]
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

for std in stds:
    pca_mse = 0
    ica_mse = 0
    nmf_mse = 0
    rnd_mse = 0

    for i in range(num_datasets):
        profiles = gen_profiles(num_celltypes, num_genes, num_markers, noise, std)
        samples, proportions = gen_samples(profiles, num_samples, std)
        # FEATURE SELECTION USING CONCRETE AUTOENCODER

        # preprocessing
        samples_normalized = normalize(samples, axis=0).T
        samples_normalized -= np.min(samples_normalized)

        # PCA
        pca = PCA(n_components=num_celltypes)
        X_pca = pca.fit_transform(samples_normalized)
        # normalize result
        n = X_pca - np.expand_dims(np.amin(X_pca, axis=1), axis=1)
        res = n / np.expand_dims(np.sum(n, axis=1), axis=1)
        print(np.shape(res))
        pca_mse += mean_squared_error(proportions, res)

        # ICA
        ica = FastICA(n_components=num_celltypes)
        X_ica = ica.fit_transform(samples_normalized)
        n = X_ica - np.expand_dims(np.amin(X_ica, axis=1), axis=1)
        res = n / np.expand_dims(np.sum(n, axis=1), axis=1)
        ica_mse += mean_squared_error(proportions, res)

        # NMF
        model = NMF(n_components=num_celltypes)
        W = model.fit_transform(samples_normalized)
        res = W / np.expand_dims(np.sum(W, axis=1), axis=1)
        nmf_mse  += (mean_squared_error(proportions, res))
        
        # Random proportions
        rnd_prop = np.random.uniform(0,1,(num_samples,num_celltypes))
        rndsum = np.sum(rnd_prop,1)
        rndsum = np.matrix([rndsum, rndsum, rndsum]).T
        rnd_prop = rnd_prop/rndsum
        rnd_mse += mean_squared_error(proportions, rnd_prop)

    # average over all datasets
    pca_mse_all.append(pca_mse/num_datasets)
    ica_mse_all.append(ica_mse/num_datasets)
    nmf_mse_all.append(nmf_mse/num_datasets)
    rnd_mse_all.append(rnd_mse/num_datasets)

plt.figure()
plt.plot(stds, pca_mse_all, label='pca')
plt.plot(stds, ica_mse_all, label='ica')
plt.plot(stds, nmf_mse_all, label='nmf')
plt.plot(stds, rnd_mse_all, label = 'random')
plt.xlabel("Standard deviation")
plt.ylabel("MSE")
#plt.ylim(0, 0.15)
plt.title("MSE per method, " + str(num_markers) + " marker genes")
plt.legend()
plt.show()