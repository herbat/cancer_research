import os, sys, math, pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

pickle_expr = 'gene_expressions.p'
pickle_meth = 'methylations.p'
pickle_expr_names = 'expr_names.p'
pickle_meth_names = 'meth_names.p'

def read_data(type='expr'):
    if type == 'expr':
        if os.path.isfile(pickle_expr) and os.path.isfile(pickle_expr_names):
            print('Loading expression pickle')
            return pickle.load(open(pickle_expr, 'rb')), pickle.load(open(pickle_expr_names, 'rb'))
        else:
            print('Reading data...')
            all_samples = []
            names = []
            c = 0
            for subdir, dirs, files in os.walk('data'):

                for file in files:
                    if file[-8:] == "FPKM.txt":
                        c += 1
                        sample = []
                        sys.stdout.write('\r')
                        # the exact output you're looking for:
                        sys.stdout.write("[%-55s] %d%%" % ('=' * math.floor(c/10), c/5.5))
                        sys.stdout.flush()
                        with open(os.path.join(subdir, file)) as f:
                            for line in f:
                                gene, exp = line.split('\t')
                                if len(names) < len(file):
                                    names.append(gene)
                                exp = float(exp.strip())
                                sample.append(exp)
                        all_samples.append(np.array(sample))
            print(np.shape(all_samples))
            final = np.vstack(all_samples)
            pickle.dump(final, open(pickle_expr, 'wb'))
            pickle.dump(names, open(pickle_expr_names, 'wb'))
            return final, names
    elif type == 'meth':
        if os.path.isfile(pickle_meth and os.path.isfile(pickle_meth_names)):
            print('Loading methylation pickle')
            return pickle.load(open(pickle_meth, 'rb')), pickle.load(open(pickle_meth_names, 'rb'))
        else:
            df = pd.read_csv(open('methylation/processed_methylation.csv', 'r', encoding='latin-1'))
            final = []
            names = []
            for l in df.to_numpy()[:-2]:
                l = l[0].split()
                names.append(l[1])
                l = l[1:]
                for i in range(498 - len(l)):
                    l.append(0.0)
                final.append(l)
            final = np.vstack(final)
            pickle.dump(final, open(pickle_meth, 'wb'))
            pickle.dump(names, open(pickle_meth_names, 'wb'))
            return final.T, names

def gen_profiles(num_profiles, genes, n_markers, noise, std, m_amp=3):
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


def test_on_sim_data():
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
            if np.min(samples_normalized) < 0:
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

            nmf_mse  += (mean_squared_error(proportions, res))

            # Random proportions
            rnd_prop = np.random.rand(num_celltypes,num_samples)
            rnd_prop /= np.sum(rnd_prop, axis = 0)
            rnd_mse += mean_squared_error(proportions, rnd_prop.T)

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
    plt.title("MSE per method, " + str(num_markers) + " marker genes, " + str(num_celltypes) + " cell types")
    plt.legend()
    plt.show()


meth_data, meth_names = read_data('meth')
expr_data, expr_names = read_data('expr')
print(meth_names, expr_names)
model = NMF(n_components=5)
print('Computing methylation proportions')
W = model.fit_transform(meth_data.T)
meth_res = W / np.expand_dims(np.sum(W, axis=1), axis=1)
print('Computing expression proportions')
W = model.fit_transform(expr_data)
expr_res = W / np.expand_dims(np.sum(W, axis=1), axis=1)
# print(mean_squared_error(expr_res, meth_res))
