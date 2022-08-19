from sklearn import svm
from sklearn.metrics.pairwise import euclidean_distances,check_pairwise_arrays,safe_sparse_dot
from feature_extractor import FeatureExtractor
from PIL import Image
from pathlib import Path
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template,url_for

def build_kernel(gamma,gamma1,coeff,degree,sigma):
    def my_kernel(X,Y):
        #X_norm = np.sum(x ** 2, axis = -1)
        #return np.exp(-(1/(2*(0.0001**2))) * (euclidean_distances(x)**2))
        #return np.exp(-(1/(2*(0.0001**2))) * (X_norm[:,None] + X_norm[None,:] - 2 * np.dot(x, x.T)))
        X, Y = check_pairwise_arrays(X, Y)
        #gamma = 1.0 / X.shape[1]
        #gamma = (1/(2*(0.0001**2)))
        #gamma = 0.0001
        #gamma1 = 0.01
        #coeff = 0
        #degree = 3
        K = euclidean_distances(X, Y, squared=True)
        K *= -gamma
        np.exp(K, K)  # exponentiate K in-place

        L = safe_sparse_dot(X,Y.T,dense_output=True)
        L*=gamma1
        L+=coeff
        L**=degree
        return (sigma*K + (1-sigma)*L)
    return my_kernel

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/Image features").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/images") / (feature_path.stem + ".jpg"))
features = np.array(features)


clf = svm.OneClassSVM(kernel=build_kernel(0.00001,0.1,0,3,0.8),nu=0.1)
#clf = svm.OneClassSVM(kernel="rbf",gamma=200,nu=0.1)
#clf = svm.OneClassSVM(kernel="poly",degree=6,gamma=0.1,nu=0.1)
clf.fit(features)
d1 = clf.decision_function(features)
r = d1.shape
dists = np.zeros(r)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        query1 = query.reshape(1,-1)
        d2 = clf.decision_function(query1)
        k = 0
        for i in d1:
            dists[k]=np.abs(d2-i)
            k=k+1
        #print(dists)
        ids = np.argsort(dists)[:20]  # Top 20 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)
