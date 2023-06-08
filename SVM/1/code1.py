import os
import numpy as np
from sklearn import svm
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

n_samples = 200
X, y = make_blobs(n_samples=n_samples, centers=2,random_state=0,cluster_std=0.7)

# kernel = "linear"
# kernel = "poly"
# kernel = "rbf"
kernel = "sigmoid"
# clf = svm.SVC(kernel=kernel ,degree=5, gamma=0.1)#poly
clf = svm.SVC(kernel=kernel , gamma="auto")# gamma =scale, auto ->poly,rbf,sigmoid
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

ax.contour(
    XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
)
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,linewidth=1,facecolors="none",edgecolors="k",
)

ax.set_title('SVM classification')

directory = os.path.dirname(os.path.abspath(__file__))

name = "/kernel=" + str(kernel) +  " samples=" + str(n_samples) + ".png"

save_path = directory + "/results" + name
print(save_path)
plt.savefig(save_path)
plt.show()