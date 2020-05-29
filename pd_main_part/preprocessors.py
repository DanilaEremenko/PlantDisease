import numpy as np


class PreprocessorInterface():
    def preprocess(self, x):
        raise Exception('Method segment need to be overrided')


###########################################################################
# ---------------------- realizations -------------------------------------
###########################################################################
class UnetPreprocessor(PreprocessorInterface):
    def __init__(self, json_path, h5_path):
        from pd_lib.keras_addition_ import get_full_model
        self.model = get_full_model(json_path=json_path, h5_path=h5_path)

    def preprocess(self, x):
        return self.model.predict(x)


class EMPreprocessor(PreprocessorInterface):
    def __init__(self, clusters_n=2):
        self.clusters_n = clusters_n

    def preprocess(self, x):
        import cv2
        def getsamples(img):
            x, y, z = img.shape
            samples = np.empty([x * y, z])
            index = 0
            for i in range(x):
                for j in range(y):
                    samples[index] = img[i, j]
                    index += 1
            return samples

        output = x.copy()
        colors = np.array([[255, 0, 0], [0, 0, 0]])
        samples = getsamples(x)
        em = cv2.ml.EM_create()
        em.setClustersNumber(self.clusters_n)
        em.trainEM(samples)
        means = em.getMeans()
        covs = em.getCovs()  # Known bug: https://github.com/opencv/opencv/pull/4232
        distance = [0] * self.clusters_n
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(self.clusters_n):
                    diff = x[i, j] - means[k]
                    distance[k] = abs(np.dot(np.dot(diff, covs[k]), diff.T))
                output[i][j] = colors[distance.index(max(distance))]
        return output


class ContrastPreprocessor(PreprocessorInterface):
    def __init__(self, k=1.5):
        self.k = k

    def preprocess(self, x):
        return x / self.k


###############################################################################
# ------------------------ get ------------------------------------------------
###############################################################################
def get_preprocessor_by_name(name, args):
    if name.lower() == 'unet':
        return UnetPreprocessor(**args)
    elif name.lower() == 'em':
        return EMPreprocessor(**args)
    elif name.lower() == 'contrast':
        return ContrastPreprocessor(**args)
    else:
        raise Exception('Undefined segmentator name = %s' % name)
