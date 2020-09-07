import numpy as np
from predict import load_test_data
from unetdeeplidate import preprocess
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score


path = 'D:/18zhuhaipeng/ADIDC-Net/Covid/9229samples/change_original_result4_bz=32_dp=0.2_2dense_again/raw/file/lr-5-32-100/'
predicted_masks = np.load( path + 'predict.npy')
imgs_test, imgs_test_mask = load_test_data()
imgs_test_gt = preprocess(imgs_test_mask)
predicted_masks_flat = predicted_masks.flatten()
test_gt_masks_flat = imgs_test_gt.flatten()


#Area under the ROC curve
fpr, tpr, thresholds = roc_curve(test_gt_masks_flat, predicted_masks_flat, pos_label=255)
auc=metrics.auc(fpr,tpr)
print("auc:",auc)
import matplotlib.pyplot as plt
#plt.plot(list(fpr),list(tpr))
plt.plot([0,1],[0,1],'k--')
line1, = plt.plot(fpr,tpr,'b',label='ADIDC-ROC (AUC = %0.4f)' % auc)

plt.legend(handles=[line1],loc=4,prop={'size':12})
#plt.plot(list(fpr),list(tpr))
plt.xlim(0,1.0)
plt.ylim(0,1.0)
plt.xlabel("False Positive Rate (1-Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.grid()
plt.savefig(path+'roc')
# plt.show()




#Precision-recall curve
precision, recall, thresholds = precision_recall_curve(test_gt_masks_flat, predicted_masks_flat, pos_label=255)
precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
AUC_prec_rec = np.trapz(precision,recall)
print("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))
prec_rec_curve = plt.figure()
plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
plt.title('Precision - Recall curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig(path+"Precision_recall.png")


