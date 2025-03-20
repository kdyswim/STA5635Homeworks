from hmmlearn import hmm
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir(os.getcwd())
data = pd.read_csv('Data/hmm_pb1.csv', header = None)
data = data.values.flatten() - 1

## 1-(a)
model = hmm.CategoricalHMM(n_components = 2, verbose = True)
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
model.emissionprob_ = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [1/10, 1/10, 1/10, 1/10, 1/10, 1/2]])

_, y = model.decode(data.reshape(-1, 1), algorithm = 'viterbi')
print(y + 1)

## 1-(b)
def forward_and_backward(x, start, trans, emis):
    alpha = np.zeros((len(x), len(start)))
    c = np.zeros(len(x))
    for i in range(len(start)):
        alpha[0, i] = start[i] * emis[i, x.reshape(-1, 1)[0, 0]]
    c[0] = 1 / np.sum(alpha[0])
    alpha[0] *= c[0]
    for t in range(1, len(x)):
        for i in range(len(start)):
            alpha[t, i] = 0
            for j in range(len(start)):
                alpha[t, i] += alpha[t - 1, j] * trans[j, i]
            alpha[t, i] *= emis[i, x.reshape(-1, 1)[t, 0]]
        c[t] = 1 / np.sum(alpha[t])
        alpha[t] *= c[t]

    beta = np.zeros((len(x), len(start)))
    beta[len(x) - 1, :] = 1
    beta[len(x) - 1, :] *= c[len(x) - 1]

    for t in range(len(x) - 2, -1, -1):
        for i in range(len(start)):
            beta[t, i] = 0
            for j in range(len(start)):
                beta[t, i] += trans[i, j] * emis[j, x.reshape(-1, 1)[t + 1, 0]] * beta[t + 1, j]
        beta[t, :] *= c[t]

    return alpha, beta, c

alpha, beta, c = forward_and_backward(data, model.startprob_, model.transmat_, model.emissionprob_)
print(f'Alpha ratio : {alpha[123, 0] / alpha[123, 1]}')
print(f'Beta ratio : {beta[123, 0] / beta[123, 1]}')

## 2
data = pd.read_csv('Data/hmm_pb2.csv', header = None)
data = data.values.flatten() - 1

def estep(x, alpha, beta, trans, emis):
    xi = np.zeros((len(x) - 1, len(trans), len(trans)))
    gamma = np.zeros((len(x), len(trans)))

    for i in range(len(x)):
        for j in range(len(trans)):
            gamma[i, j] = alpha[i, j] * beta[i, j]
        gamma[i] /= np.sum(gamma[i])

    for i in range(len(x) - 1):
        den = 0
        for j in range(len(trans)):
            for k in range(len(trans)):
                xi[i, j, k] = alpha[i, j] * trans[j, k] * emis[k, x.reshape(-1, 1)[i + 1, 0]] * beta[i + 1, k]
                den += xi[i, j, k]

        if den > 0:
            xi[i] /= den

    return xi, gamma

start = np.array([0.5, 0.5])
trans = np.array([[0.75, 0.25], [0.25, 0.75]])
emis = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [1/10, 1/10, 1/10, 1/10, 1/10, 1/2]])

n_iter = 300
for iter in range(n_iter):
    alpha, beta, c = forward_and_backward(data, start, trans, emis)
    xi, gamma = estep(data, alpha, beta, trans, emis)

    start = gamma[0]
    for i in range(len(trans)):
        den = np.sum(gamma[:-1, i])
        for j in range(len(trans)):
            num = np.sum(xi[:, i, j])
            trans[i, j] = num / den if den > 0 else 0

    for i in range(len(trans)):
        den = np.sum(gamma[:, i])
        for j in range(len(emis[0])):
            ind = np.where(data.reshape(-1, 1)[:, 0] == j)[0]
            num = np.sum(gamma[ind, i]) if len(ind) > 0 else 0
            emis[i, j] = num / den if den > 0 else 0

    if iter == (n_iter - 1):
        print('Initial :', start)
        print('Trans :', trans)
        print('Emis :', emis)