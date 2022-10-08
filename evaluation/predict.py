import numpy as np

def evaluate(model_fit, tensor_1, tensor_0, threshold=0.5):

  y_pred1 = []
  for t in tensor_1:

    y_pred1.append(1 if model_fit.predict_step(t) > threshold else 0)

  NP = len(y_pred1)
  TP = sum(y_pred1)
  FN = NP - TP

  y_pred0 = []
  for t in tensor_0:
    y_pred0.append(1 if model_fit.predict_step(t) > threshold else 0)

  NN = len(y_pred0)
  FP = sum(y_pred0)
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

def predict(model_fit, tensor, threshold=0.5):
  y_pred = []
  for t in tensor:
    y_pred.append(1 if model_fit.predict_step(t) > threshold else 0)

  return np.array(y_pred)