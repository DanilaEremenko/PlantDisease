from pd_lib.keras_addition_ import get_full_model


class ClassifierInterface():
    def predict(self, x_arr):
        raise Exception('Method predict need to be overrided')


###########################################################################
# ---------------------- realizations -------------------------------------
###########################################################################
class CNNClassifier(ClassifierInterface):
    def __init__(self, json_path, h5_path, classes):
        self.model = get_full_model(json_path=json_path, h5_path=h5_path)
        self.classes = classes

    def predict(self, x_arr):
        return self.model.predict(x_arr)


###############################################################################
# ------------------------ get ------------------------------------------------
###############################################################################
def get_classifier_by_name(name, args):
    if name.lower() == 'cnn':
        return CNNClassifier(**args)
    else:
        raise Exception('Undefined cluster name = %s' % name)
