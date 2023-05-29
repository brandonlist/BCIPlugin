from PluginCore.Decoder.base import Inspector

class SkorchInspector(Inspector):
    def __init__(self, inspector, cuda):
        self.inspector = inspector
        self.cuda = cuda

    def inspect(self, test_X, test_y, model):
        module = model.module

        if self.cuda:
            test_X = test_X.cuda()
            test_y = test_y.cuda()

        y_true = test_y.cpu()
        y_pred = model.predict(test_X)
        y_logit = module(test_X).detach().cpu()
        re = self.inspector.inspect(y_true,y_pred,y_logit)
        return re

