class SegmentatorInterface():
    def segment(self, x_arr):
        raise Exception('Method segment need to be overrided')


###########################################################################
# ---------------------- realizations -------------------------------------
###########################################################################
class UnetSegmentator(SegmentatorInterface):
    def __init__(self, json_path, h5_path):
        from pd_lib.keras_addition_ import get_full_model
        self.model = get_full_model(json_path=json_path, h5_path=h5_path)

    def segment(self, x_arr):
        return self.model.predict(x_arr)


class EMSegmentator(SegmentatorInterface):
    def __init__(self, clusters_n):
        self.clusters_n = clusters_n

    def EMSegmentation(self, img):
        import cv2
        import numpy as np
        def getsamples(img):
            x, y, z = img.shape
            samples = np.empty([x * y, z])
            index = 0
            for i in range(x):
                for j in range(y):
                    samples[index] = img[i, j]
                    index += 1
            return samples

        output = img.copy()
        colors = np.array([[255, 0, 0], [0, 0, 0]])
        samples = getsamples(img)
        em = cv2.ml.EM_create()
        em.setClustersNumber(self.clusters_n)
        em.trainEM(samples)
        means = em.getMeans()
        covs = em.getCovs()  # Known bug: https://github.com/opencv/opencv/pull/4232
        x, y, z = img.shape
        distance = [0] * self.clusters_n
        for i in range(x):
            for j in range(y):
                for k in range(self.clusters_n):
                    diff = img[i, j] - means[k]
                    distance[k] = abs(np.dot(np.dot(diff, covs[k]), diff.T))
                output[i][j] = colors[distance.index(max(distance))]
        return output


###############################################################################
# ------------------------ get ------------------------------------------------
###############################################################################
def get_segmentator_by_name(name, args):
    if name.lower() == 'unet':
        return UnetSegmentator(**args)
    elif name.lower() == 'em':
        return EMSegmentator(**args)
    else:
        raise Exception('Undefined segmentator name = %s' % name)
