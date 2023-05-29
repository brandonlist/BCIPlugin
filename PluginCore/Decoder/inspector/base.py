from sklearn.metrics import classification_report,precision_recall_fscore_support,confusion_matrix,cohen_kappa_score,\
    accuracy_score,precision_score,recall_score,f1_score,fbeta_score,precision_recall_curve,average_precision_score,\
    roc_curve,auc,roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
from copy import deepcopy
eps = 1e-6

def ITR_wolpaw(y_true, y_pred):
    """

    Reference: EEG-based communication: improved accuracy by response verification
    :param y_true:
    :param y_pred:
    :return:
    """

    acc = accuracy_score(y_true, y_pred)
    M = len(np.unique(y_true))
    ITR = np.log2(M) + acc * np.log2(acc + eps) + (1-acc) * np.log2((1-acc)/(M-1) + eps)
    return ITR

def ITR_nykopp(y_true, y_pred):
    """

    Reference: Statistical modelling issues for the adaptive brain interface
    :param y_true:
    :param y_pred:
    :return:
    """
    cm = confusion_matrix(y_true, y_pred)
    p_y_xs = np.zeros((cm.shape[0],cm.shape[1]))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            p_y_xs[i][j] = cm[i][j] / (np.sum(cm[i]) + eps)     #p(Yj|Xi) corresponds to p_y_xs[i][j]
    p_ys = np.zeros((cm.shape[0]))
    for j in range(cm.shape[1]):
        for i in range(cm.shape[0]):
            p_ys[j] += (np.sum(cm[i]) / np.sum(cm)) * p_y_xs[i][j]
    ans = 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ans += p_y_xs[i][j] * (np.sum(cm[i]) / np.sum(cm)) * np.log2(p_y_xs[i][j] + eps)

    ans -= np.sum(p_ys * np.log2(p_ys) + eps)
    return ans

def MPM(y_true, y_pred):
    """

    Misclassification probability matrix
    :param y_true:
    :param y_pred:
    :return:
    """
    cm = confusion_matrix(y_true,y_pred)
    mpm = np.zeros_like(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i==j:
                mpm[i][j] = 0
            else:
                mpm[i][j] = cm[i][j] / np.sum(cm[i])
    return mpm

def efficiency(y_true, y_pred, OM=None, T=1):
    """

    Efficiency matrix, T is 1 by default

    ST: supertax
    OM: overtime matrix
    ESC: estimate mean selection cost
    Reference: Evaluation of the performances of different P300 based brain-computer interfaces by means of the efficiency metric
    :param y_true:
    :param y_pred:
    :param OM:
    :return:
    """
    mpm = MPM(y_true,y_pred)
    ST = np.zeros(mpm.shape[0])
    if OM is None:
        OM = np.zeros((mpm.shape[0],mpm.shape[1]))
        for i in range(mpm.shape[0]):
            for j in range(mpm.shape[1]):
                if i==j:
                    OM[i][j] = 0
                else:
                    OM[i][j] = 1
    for i in range(mpm.shape[0]):
        ST[i] = np.sum(mpm[i] * OM[i])
    ESC = 0
    for i in range(mpm.shape[0]):
        ESC += 1 / (1 - ST[i] + eps)
    efcy = 1 / (ESC * T + eps)
    return efcy

def sensitivity(y_true, y_pred, pos_label):
    assert len(np.unique(y_true))==2
    assert len(np.unique(y_pred))==2
    sst = recall_score(y_true,y_pred, pos_label=pos_label)
    return sst

def specificity(y_true, y_pred, pos_label):
    assert len(np.unique(y_true)) == 2
    assert len(np.unique(y_pred)) == 2
    classes = np.unique(y_true)
    neg_lable = [x for x in classes if x is not pos_label]
    neg_lable = neg_lable[0]
    spc = recall_score(y_true,y_pred,pos_label=neg_lable)
    return spc

def hit_false(y_true,y_pred,ic_label):
    assert len(np.unique(y_true)) == 2
    assert len(np.unique(y_pred)) == 2
    classes = np.unique(y_true)
    nc_label = [x for x in classes if x is not ic_label]
    HF = sensitivity(y_true,y_pred,pos_label=ic_label) + precision_score(y_true,y_pred,pos_label=ic_label) - 1
    return HF

def is_balanced(y,th=0.05):
    classes_ratio = [len(y[y == i]) / len(y) for i in np.unique(y)]
    return np.std(classes_ratio) < th

class InspectorStandard():
    def __init__(self,acc_normalize=True,p_average='micro',gotNames=False,
                 r_average='micro',f1_average='micro',fb_average='micro',beta=1,
                 pr_average='micro',roc_average='macro',manual_n_classes=None):
        self.parameters = {}
        self.parameters['gotNames'] = gotNames
        self.parameters['acc_normalize'] = acc_normalize
        self.parameters['p_average'] = p_average
        self.parameters['r_average'] = r_average
        self.parameters['f1_average'] = f1_average
        self.parameters['fb_average'] = fb_average
        self.parameters['beta'] = beta
        self.parameters['pr_average'] = pr_average
        self.parameters['roc_average'] = roc_average
        self.manual_n_classes = manual_n_classes

    def inspect(self, y_true, y_pred, y_logit=None):
        if self.manual_n_classes is None:
            self.n_classes = len(np.unique(y_true))
        else:
            self.n_classes = self.manual_n_classes
        self.n_samples = len(y_true)
        assert len(y_true)==len(y_pred)
        results = {}

        #Metrics valid for both Bi-class and Multi-class scenario
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred, normalize=self.parameters['acc_normalize'])
        kappa = cohen_kappa_score(y_true, y_pred)

        itr_wolpaw = ITR_wolpaw(y_true,y_pred)
        itr_nykopp = ITR_nykopp(y_true,y_pred)
        results['cm'] = cm ;results['acc'] = acc;results['kappa'] = kappa;results['error']=1-acc
        results['itr_wolpaw'] = itr_wolpaw; results['itr_nykopp'] = itr_nykopp

        #Metrics that have different extension methods to multi class
        p = precision_score(y_true, y_pred, average=self.parameters['p_average'])
        r = recall_score(y_true, y_pred, average=self.parameters['r_average'])
        f1 = f1_score(y_true, y_pred, average=self.parameters['f1_average'])
        fbeta = fbeta_score(y_true,y_pred,beta=self.parameters['beta'],average=self.parameters['fb_average'])
        results['p'] = p;results['r'] = r;results['f1'] = f1;results['fbeta'] = fbeta

        # if type(y_logit)==np.ndarray:
        if y_logit.ndim==2:
            if self.n_classes==2:
                y_one_hot = label_binarize(y_true, np.arange(self.n_classes+1))
                y_one_hot = np.delete(y_one_hot,-1,axis=1)
            else:
                y_one_hot = label_binarize(y_true, np.arange(self.n_classes))
            if self.parameters['pr_average'] == 'micro':
                p_curve, r_curve, th_pr = precision_recall_curve(y_one_hot.ravel(), y_logit.ravel())
                average_precision = average_precision_score(y_one_hot, y_logit, average='micro')
            elif self.parameters['pr_average'] == 'macro':
                p_curve, r_curve ,th_pr = [],[],[]
                ap = 0
                for i in range(self.n_classes):
                    y_logit_i = y_logit[:, i]
                    y_one_hot_i = y_one_hot[:, i]
                    p_curve_i, r_curve_i, th_pr_i = precision_recall_curve(y_one_hot_i, y_logit_i)
                    # 注意这里每次算出的fpr_i和tpr_i形状都有可能不同
                    p_curve.append(p_curve_i);r_curve.append(r_curve_i);th_pr.append(th_pr_i)
                    ap_i = average_precision_score(y_one_hot_i, y_logit_i)
                    ap += ap_i
                ap /= y_one_hot.shape[1] #ap==average_precision
                average_precision = average_precision_score(y_one_hot, y_logit, average='macro')
            results['p_curve'] = p_curve;results['r_curve'] = r_curve
            results['th_pr'] = th_pr;results['average_precision'] = average_precision


            if self.parameters['roc_average'] == 'micro':
                fpr_curve, tpr_curve, th_roc = roc_curve(y_one_hot.ravel(), y_logit.ravel())
                roc_auc = roc_auc_score(y_one_hot,y_logit,average='micro')
            elif self.parameters['roc_average'] == 'macro':
                fpr_curve, tpr_curve, th_roc = [], [], []
                roc_auc = 0
                for i in range(y_one_hot.shape[1]):
                    y_logit_i = y_logit[:, i]
                    y_one_hot_i = y_one_hot[:, i]
                    fpr_i, tpr_i, th_roc_i = roc_curve(y_one_hot_i, y_logit_i)
                    # 注意这里每次算出的fpr_i和tpr_i形状都有可能不同
                    fpr_curve.append(fpr_i);tpr_curve.append(tpr_i);th_roc.append(th_roc_i)
                    ac_i = auc(fpr_i, tpr_i)
                    roc_auc += ac_i
                roc_auc /= self.n_classes
            results['fpr_curve'] = fpr_curve;results['tpr_curve'] = tpr_curve
            results['th_roc'] = th_roc;results['roc_auc'] = roc_auc

        del self.n_classes,self.n_samples
        return results

class InspectorSyn(InspectorStandard):
    def __init__(self,pos_label,manual_n_classes=None,
                 acc_normalize=True,p_average='micro',gotNames=False,
                 r_average='micro',f1_average='micro',fb_average='micro',beta=1,
                 pr_average='micro',roc_average='macro',):
        super(InspectorSyn, self).__init__(acc_normalize=acc_normalize,p_average=p_average,gotNames=gotNames,
                 r_average=r_average,f1_average=f1_average,fb_average=fb_average,beta=beta,
                 pr_average=pr_average,roc_average=roc_average,manual_n_classes=manual_n_classes)
        self.pos_label = pos_label

    def inspect(self, y_true, y_pred, y_logit=None):
        results = super(InspectorSyn, self).inspect(y_true=y_true,y_pred=y_pred,y_logit=y_logit)
        results_ret = deepcopy(results)
        n_classes = len(np.unique(y_true))
        if n_classes==2:
            if is_balanced(y_true):
                results_ret['BCI_acc'] = results['acc']
                results_ret['BCI_itr_wolpaw'] = results['itr_wolpaw']
            else:
                results_ret['BCI_kappa'] = results['kappa']
                results_ret['BCI_itr_nykopp'] = results['itr_nykopp']

            if not is_balanced(y_pred):
                if self.pos_label in np.unique(y_true):
                    results_ret['BCI_cm'] = results['cm']
                    results_ret['BCI_sensitivity'] = sensitivity(y_true,y_pred,self.pos_label)
                    results_ret['BCI_specificity'] = specificity(y_true,y_pred,self.pos_label)
                    results_ret['BCI_precision'] = results['average_precision']
                else:
                    results_ret['BCI_cm'] = None
                    results_ret['BCI_precision'] = None
                    results_ret['BCI_sensitivity'] = None
                    results_ret['BCI_specificity'] = None



        elif n_classes>2:
            if is_balanced(y_pred):
                results_ret['BCI_acc'] = results['acc']
                results_ret['BCI_kappa'] = results['kappa']
                results_ret['BCI_itr_nykopp'] = results['itr_nykopp']
            else:
                results_ret['BCI_cm'] = results['cm']
                results_ret['BCI_precision'] = results['average_precision']
        else:
            print('number of class should be >= 2')

        return results_ret

class InspectorBrainSwitch(InspectorStandard):
    def __init__(self,ic_label,
                 acc_normalize=True,p_average='micro',gotNames=False,
                 r_average='micro',f1_average='micro',fb_average='micro',beta=1,
                 pr_average='micro',roc_average='macro',):
        super(InspectorBrainSwitch, self).__init__(acc_normalize=acc_normalize,p_average=p_average,gotNames=gotNames,
                 r_average=r_average,f1_average=f1_average,fb_average=fb_average,beta=beta,
                 pr_average=pr_average,roc_average=roc_average)
        self.ic_label = ic_label

    def inspect(self, y_true, y_pred, y_logit=None):
        assert self.ic_label in np.unique(y_true)
        results = super(InspectorBrainSwitch, self).inspect(y_true=y_true,y_pred=y_pred,y_logit=y_logit)

        results['BCI_hit_false'] = hit_false(y_true,y_pred,ic_label=self.ic_label)
        results['BCI_kappa'] = results['kappa']
        results['BCI_sensitivity'] = sensitivity(y_true,y_pred,pos_label=self.ic_label)
        results['BCI_specificity'] = specificity(y_true,y_pred,pos_label=self.ic_label)
        results['BCI_precision'] = results['average_precision']

        return results
