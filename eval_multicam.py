import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

num_cameras = 8
L = 10
num_features = 4096
batch_norm = True
learning_rate = 0.01
mini_batch_size = 0
weight_0 = 1
epochs = 6000
use_validation = False
# After the training stops, use train+validation to train for 1 epoch
use_val_for_training = False
val_size = 100
# Threshold to classify between positive and negative
threshold = 0.5

sensitivities = []
specificities = []
aucs = []
accuracies = []

for cam in range(24):
    
    # ==================== EVALUATION ========================  
    # Load best model
    predicted = []
    y_test = None
    for i in range(len(predicted)):
        if predicted[i] < threshold:
            predicted[i] = 0
        else:
            predicted[i] = 1
    # Array of predictions 0/1
    predicted = np.asarray(predicted).astype(int)
    
    # Compute metrics and print them
    cm = confusion_matrix(y_test, predicted,labels=[0,1])
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]
    tpr = tp/float(tp+fn)
    fpr = fp/float(fp+tn)
    fnr = fn/float(fn+tp)
    tnr = tn/float(tn+fp)
    precision = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    specificity = tn/float(tn+fp)
    f1 = 2*float(precision*recall)/float(precision+recall)
    accuracy = accuracy_score(y_test, predicted)
    fpr, tpr, _ = roc_curve(y_test, predicted)
    roc_auc = auc(fpr, tpr)
    
    print('FOLD/CAMERA {} results:'.format(cam))
    print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
    print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(
                    tpr,tnr,fpr,fnr))   
    print('Sensitivity/Recall: {}'.format(recall))
    print('Specificity: {}'.format(specificity))
    print('Precision: {}'.format(precision))
    print('F1-measure: {}'.format(f1))
    print('Accuracy: {}'.format(accuracy))
    print('AUC: {}'.format(roc_auc))
    
    # Store the metrics for this epoch
    sensitivities.append(tp/float(tp+fn))
    specificities.append(tn/float(tn+fp))
    aucs.append(roc_auc)
    accuracies.append(accuracy)
    
print('LEAVE-ONE-OUT RESULTS ===================')
print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities),
                        np.std(sensitivities)))
print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(specificities),
                        np.std(specificities)))
print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies),
                        np.std(accuracies)))
print("AUC: %.2f%% (+/- %.2f%%)" % (np.mean(aucs), np.std(aucs)))