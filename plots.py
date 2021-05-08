import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

matplotlib.rc('xtick', labelsize=22)
matplotlib.rc('ytick', labelsize=22)
plt.rcParams.update({'font.size': 22})


def plot_roc_handy(y_test, y_score, lw=2, name='Roc', class_name=['COVID19', 'Normal', 'Pneumonia'], zoom=False,
                   axis=[0.0, 0.12, 0.88, 1.0]):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(int(y_test.shape[1])):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(int(y_test.shape[1]))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(int(y_test.shape[1])):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= int(y_test.shape[1])
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    f, ax = plt.subplots(figsize=[15, 15])

    ax.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    ax.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(int(y_test.shape[1])), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                      ''.format(class_name[i], roc_auc[i]))
    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    ax.set_xlim([0.001, 1.0])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=22)
    ax.set_ylabel('True Positive Rate', fontsize=22)
    ax.set_title(name, fontsize=22)
    ax.legend(loc="lower right")

    # inset axes....
    if zoom:
        axins = ax.inset_axes([0.3, 0.4, 0.4, 0.4])
        for i, color in zip(range(int(y_test.shape[1])), colors):
            if fpr[i].all() < 0.2 and tpr[i].all() < 0.95:
                axins.plot(fpr[i], tpr[i], color=color, lw=lw,
                           label='ROC curve of class {0} (area = {1:0.2f})'
                                 ''.format(class_name[i], roc_auc[i]))

                axins.plot(fpr["micro"], tpr["micro"],
                           label='micro-average ROC curve (area = {0:0.2f})'
                                 ''.format(roc_auc["micro"]),
                           color='deeppink', linestyle=':', linewidth=4)

                axins.plot(fpr["macro"], tpr["macro"],
                           label='macro-average ROC curve (area = {0:0.2f})'
                                 ''.format(roc_auc["macro"]),
                           color='navy', linestyle=':', linewidth=4)

        x1, x2, y1, y2 = axis
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels('')
        axins.set_yticklabels('')
        ax.indicate_inset_zoom(axins)

    plt.show()
    f.savefig('{}.pdf'.format(name))
    ax.figure.savefig("{}.pdf".format(name), bbox_inches='tight')


def plot_cm_handy(y_test, y_score, lw=2, name='Confusion Matrix of Fusion Model without Uncertainty (X-Ray)',
                  class_name=['COVID19', 'Normal', 'Pneumonia']):

    CM = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_score, axis=1))
    cm = CM
    cmap = plt.cm.Blues
    fig, ax = plt.subplots(figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(name)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_name, yticklabels=class_name,
           ylabel='True Label',
           xlabel='Predicted Label')
    ax.set_xticklabels(class_name, fontsize=15)
    ax.set_yticklabels(class_name, fontsize=15)
    ax.set_ylabel('True Label', fontsize=15)
    ax.set_xlabel('Predicted Label', fontsize=15)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            figure(num=None, figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
            ax.text(j, i, format(cm[i, j], fmt), fontsize=12,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig('{}.pdf'.format(name), dpi=300)
    ax.figure.savefig("{}.pdf".format(name), bbox_inches='tight')
