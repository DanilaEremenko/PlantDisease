from pd_lib.data_maker import get_x_from_croped_img


class ClustererInterface():
    def cluster(self, path_to_img):
        raise Exception('Method predict need to be overrided')


###########################################################################
# ---------------------- realizations -------------------------------------
###########################################################################
class SlidingWindow(ClustererInterface):
    def __init__(self, window_shape, step, img_thumb):
        self.window_shape = window_shape
        self.step = step
        self.img_thumb = img_thumb

    def cluster(self, path_to_img):
        result_dict, full_img = get_x_from_croped_img(
            path_to_img=path_to_img,
            window_shape=self.window_shape,
            step=self.step,
            img_thumb=self.img_thumb,
            verbose=False
        )

        # TODO add logic with color checking

        return result_dict


###############################################################################
# ------------------------ get ------------------------------------------------
###############################################################################
def get_clusterer_by_name(name, args):
    if name.lower() == 'slidingwindow':
        return SlidingWindow(**args)
    else:
        raise Exception('Undefined cluster name = %s' % name)
