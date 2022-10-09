import numpy as np

def evaluate(model_fit,
            tensors_silos,
            tensor_pas_silos,
            threshold=0.5):

  y_pred_on_silos = [1 if model_fit.predict_step(t) > threshold else 0 for t in tensor_silos]
  
  NP = len(y_pred_on_silos)
  TP = sum(y_pred_on_silos)
  FN = NP - TP

  y_pred_on_pas_silos = [1 if model_fit.predict_step(t) > threshold else 0 for t in tensor_pas_silos]

  NN = len(y_pred_on_pas_silos)
  FP = sum(y_pred_on_pas_silos)
  TN = NN - FN

  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  F1 = (2*precision*recall) / (precision + recall)

  TPR = TP / (TP + FN)
  FNR = FN / (TP + FN)
  TNR = TN / (TN + FP)
  FPR = FP / (TN + FP)

  return {"confusion_matrix" : np.array([[TP, FP],[FN, TN]]),
          "precision" : precision,
          "recall" : recall, 
          "F1-score" : F1, 
          "TPR" : TPR,
          "FNR" : FNR,
          "TNR" : TNR,
          "FPR" : FPR
        }

def predict(model_fit,
tensor,
threshold=0.5):
  y_pred = []
  for t in tensor:
    y_pred.append(1 if model_fit.predict_step(t) > threshold else 0)

  return np.array(y_pred)