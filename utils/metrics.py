#Coded by Jinhua Liang in Tianjin Univercity
#=============================================================================
import numpy
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

from sklearn.metrics import confusion_matrix


EPS = 1e-10
labels = ['airport', 'bus', 'metro', 'metro st.', 'park', 'publi sq.', 
    'shopping', 'street pe.', 'street tr.', 'tram']


def plot_matrix(matrix, cmap='GnBu'):
    """plot customized norm confusion matrix in a table"""
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.colorbar(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=45)
    plt.yticks(xlocations, labels)
    # for confusion matrix y axis is label
    plt.ylabel('True label')
    # for confusion matrix x axis is prediction
    plt.xlabel('Predicted label')


def plot_confusion_matrix(eval_labels, prediction_labels, setting,
                          name='confusion matrix'):
    """Compute confusion matrix, accuracy, macro_P, macro_R and macro_F
    
    :param eval_labels: array, shape = [n_samples] Ground truth (correct) 
        target values.
    :param prediction_labels: array, shape = [n_samples] 
        Estimated targets as returned by a classifier.
    
    :return:
        C : array, shape = [n_classes, n_classes]
        confusion_matrixs: array, shape = [n_samples, n_samples]
        accuracy: float64
        macro_P: float64
        macro_R: float64
        macro_F: float64
    """
    tick_marks = np.array(range(len(labels))) + 0.5
    # calculate a normalized confustion_matrix sized 10*10
    confusion_matrixs = confusion_matrix(eval_labels, prediction_labels)
    confusion_matrixs_normalized = confusion_matrixs.astype('float') / confusion_matrixs.sum(axis=1)[:, np.newaxis]
    # print('混淆矩阵')
    # print(confusion_matrixs_normalized)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = confusion_matrixs_normalized[y_val][x_val]
        plt.text(x_val, y_val, "%0.2f" % (c,), 
            color='black', fontsize=7, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    # plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.2)
    plot_matrix(confusion_matrixs_normalized)
    plt.savefig(name + '.jpg', format='jpg', **setting)
    # plt.show()


def metrics_statitic(y_true, y_pred, n_cls=10, stat=None):
    """ Compute acc, precision, recall and f1-score
    
    ----Args----
    y_true: array, shape = [n_samples] Ground truth (correct) target values.
    y_pred: array, shape = [n_samples] Estimated targets by a classifier.
    n_cls: num of class, default is 10
    stat: return precition, recall and f-score in stat, None (default) stat per class.
    
    ----return----
        acc: float64 or list
        p: float64 or list
        r: float64 or list
        f: float64 or list
    """
    # calculate a normalized confustion_matrix sized n * n
    confusion_matrixs = confusion_matrix(y_true, y_pred)
    
    # convert the multi- metric to binaral-classification (total num = 10) 
    n = n_cls
    TP = numpy.empty(n, dtype=numpy.int64, order='F')
    TN = numpy.empty(n, dtype=numpy.int64, order='F')
    FP = numpy.empty(n, dtype=numpy.int64, order='F')
    FN = numpy.empty(n, dtype=numpy.int64, order='F')
    E = numpy.empty(n, dtype=numpy.float64, order='F')
    P = numpy.empty(n, dtype=numpy.float64, order='F')
    R = numpy.empty(n, dtype=numpy.float64, order='F')
    F = numpy.empty(n, dtype=numpy.float64, order='F')
    
    for i in range(n):
        TP[i] = confusion_matrixs[i][i]
        FP[i] = sum(confusion_matrixs[:, i]) - TP[i]
        FN[i] = sum(confusion_matrixs[i]) - TP[i]
        TN[i] = sum(sum(confusion_matrixs)) - TP[i] - FP[i] - FN[i]
        
        E[i] = (FN[i] + FP[i]) / (sum(sum(confusion_matrixs)) + EPS)
        P[i] = TP[i] / (TP[i] + FP[i] + EPS)
        R[i] = TP[i] / (TP[i] + FN[i] + EPS)
        F[i] = (2 * P[i] * R[i]) / (P[i] + R[i] + EPS)

    acc = sum(TP) / (sum(sum(confusion_matrixs)) + EPS)
    
    if stat == 'macro':
        print("MACRO metrics are:")
        err = sum(E) / n
        p = sum(P) / n
        r = sum(R) / n
        f = sum(F) / n
        
    elif stat == 'micro':
        print("micro metrics are:")
        err = (sum(FN) + sum(FP))/ (n * sum(sum(confusion_matrixs)) + EPS)
        p = sum(TP) / (sum(TP) + sum(FP) + EPS)
        r = sum(TP) / (sum(TP) + sum(FN) + EPS)
        f = (2 * p * r) / (p + r + EPS)
        
    else:
        print("metrics in binaral-classification are")
        err = E
        p = P
        r = R
        f = F

    return acc, err, p, r, f
