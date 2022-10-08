import numpy as np

def evaluate(model_fit, tensor_1, tensor_0):
  y_pred1 = []
  for t in tensor_1:
    y_pred1.append(round(float(model_fit.predict_step(t))))

  NP = len(y_pred1)
  TP = sum(y_pred1)
  FN = NP - TP

  y_pred0 = []
  for t in tensor_0:
    y_pred0.append(round(float(model_fit.predict_step(t))))

  NN = len(y_pred0)
  FP = sum(y_pred0)
  TN = NN - FN

  precision = TP / (TP + FP)
  recall = TP / (TP + FN)

  F1 = (2*precision*recall) / (precision + recall)

  return np.array([[TP, FP],[FN, TN]]), precision, recall, F1

