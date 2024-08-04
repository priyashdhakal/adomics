import arguments as args 
import os
import logging 
logging.getLogger().setLevel(logging.INFO)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np 
from sklearn.metrics import roc_curve, auc 

plots_dir = args.PlotUtilArguments.plot_dir
if not os.path.exists(plots_dir):
    logging.info(f"Creating directory {plots_dir}")
    os.makedirs(plots_dir)

def plot_roc_curve(y_true, y_pred, save_data = True):
    if save_data:
        logging.info(f"Raw Predictions {plots_dir}")
        np.save(os.path.join(plots_dir, "raw_predictions.npy"), y_pred)
        logging.info(f"True Labels saved to {plots_dir}")
        np.save(os.path.join(plots_dir, "true_labels.npy"), y_true)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plots_dir, "roc_curve.png"))
    plt.clf()
    logging.info(f"ROC Curve saved to {plots_dir}")

    return True

def plot_history(history):
    """
    Save two different plots, 1 for loss and another for accuracy from history object 
    returned by tensorflow model.fit
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(plots_dir, "accuracy.png"))
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(plots_dir, "loss.png"))
    plt.clf()
    logging.info(f"Plots saved to {plots_dir}")
    return True

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    with open(os.path.join(plots_dir, "confusion_matrix.txt"), "w") as f:
        f.write(str(cm))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.clf()
    logging.info(f"Confusion Matrix saved to {plots_dir}")
    return True