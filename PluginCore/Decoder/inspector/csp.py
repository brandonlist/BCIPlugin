from PluginCore.Decoder.base import Inspector
import numpy as np

class CSPInspector(Inspector):
    def __init__(self, inspector):
        self.inspector = inspector

    def inspect(self, test_X, test_y, model):
        try:
            test_X = test_X.numpy().astype(np.float)
            test_y = test_y.numpy()
        except:
            pass

        y_true = test_y
        y_pred = model.predict(test_X)
        y_logit = model.logits(test_X)
        re = self.inspector.inspect(y_true, y_pred, y_logit)
        return re