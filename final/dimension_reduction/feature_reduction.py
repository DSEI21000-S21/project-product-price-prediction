from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import Isomap



def dimension_reduction(data, method, n_comp):
    random_num = int(datetime.now().timestamp())
    techniques = {
        'PCA': PCA(n_components=n_comp),
        'SVD': TruncatedSVD(n_components=n_comp, random_state=random_num),
        'LatentDA': LatentDirichletAllocation(n_components=n_comp,
                                  learning_method='online', # use mini-batch update
                                  random_state=random_num),
        'KMeans': MiniBatchKMeans(n_clusters=n_comp, init_size=1000,
                                  batch_size=500, max_iter=2000),
        'Isomap': Isomap(n_components=n_comp)
    }
    try:
        model = techniques[method]
        return model.fit_transform(data)
    except:
        print("Please select the method from: PCA, SVD, LatentDA, KMeans, Isomap ")
        return None
