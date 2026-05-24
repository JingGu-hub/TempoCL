
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from cleanlab.internal.constants import EPSILON

class TASS:
    def __init__(self, args):
        self.args = args

    def gmm_divide(self, loss, stage='first'):
        if self.args.use_gmm_divide_strategy and stage == 'first':
            scaling_factor = float(max(np.median(loss), 100 * np.finfo(np.float64).eps))
            loss = np.exp(-1 * loss / max(scaling_factor, EPSILON))
            loss = 1 - loss

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(loss)
        prob = gmm.predict_proba(loss)

        thread = 0.5

        prob = prob[:, gmm.means_.argmin()]
        pred1 = (prob > thread)
        u_ids = np.where(prob <= thread)[0]

        return pred1, prob, u_ids

    def compute_ood_distances(self, features):
        knn = NearestNeighbors(n_neighbors=min(10, features.shape[0] - 1), metric='cosine')
        knn.fit(features)
        k = knn.n_neighbors

        distances, indices = knn.kneighbors(features)

        avg_knn_distances = distances[:, :k].mean(axis=1)
        scaling_factor = float(max(np.median(avg_knn_distances), 100 * np.finfo(np.float64).eps))
        avg_knn_distances = np.exp(-1 * avg_knn_distances / max(scaling_factor, EPSILON))

        return avg_knn_distances

    def obtain_ood_scores(self, features):
        indices = np.arange(len(features))
        np.random.shuffle(indices)
        features = features[indices]

        avg_knn_distances = self.compute_ood_distances(features=features)

        avg_knn_distances = avg_knn_distances[np.argsort(indices)]

        return avg_knn_distances

    def first_sample_selection(self, loss):
        return self.gmm_divide(loss, stage='first')

    def second_sample_selection(self, features, labels, u_ids):
        ood_mask = np.ones(len(labels))

        if self.args.ood_noise_rate > 0:
            train_ood_features_scores = self.obtain_ood_scores(features[u_ids].detach().cpu().numpy())
            feature_pred, feature_prob, devide_ood_ids = self.gmm_divide(train_ood_features_scores.reshape(-1, 1), stage='second')
            devide_ood_ids = u_ids[devide_ood_ids]

            if len(devide_ood_ids) != 0:
                ood_mask[devide_ood_ids] = 0

        return ood_mask
