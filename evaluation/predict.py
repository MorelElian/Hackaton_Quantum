import numpy as np
import tensorflow as tf
#model compile. 
#create mask

def proba(model_fit,
            tensors_silos,
            tensor_pas_silos):
  
  """ Returns the scores sent by the model fitted on validation data"""
  probas_silos = [model_fit.predict_step(t) for t in tensors_silos]
  probas_pas_silos = [model_fit.predict_step(t) for t in tensors_pas_silos]

  return probas_silos, probas_pas_silos

def evaluate(probas_silos,
            probas_pas_silos,
            threshold=0.5):

  """ Evaluate global metrics on validation data, depending on the prediction of the models. 
      predictions must be split into probas_silos and probas_pas_silos
      Metrics computed :
      - precision
      - recall
      - F1-score
      - True positive rate
      - False negative rate
      - True negative rate
      - False positive rate
      - Confusion matrix  """
    
  y_pred_on_silos = [1 if p > threshold else 0 for p in probas_silos]
  
  NP = len(y_pred_on_silos)
  TP = sum(y_pred_on_silos)
  FN = NP - TP

  y_pred_on_pas_silos = [1 if p > threshold else 0 for p in probas_pas_silos]

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
threshold=0.5,
proba=False):
  y_pred = []
  for t in tensor:
    y_pred.append(1 if model_fit.predict_step(t) > threshold else 0)

  return np.array(y_pred)


def iou_metric(dataset): 

  m = tf.keras.metrics.IoU(num_classes=2,target_class_ids=[1])
  for image, mask in dataset:
    model_compiled = model_compile()
    print(type(model_compiled))
    model_compiled.load_weights('weights100.hdf5')
    
    predicted_image = model_compiled.predict(image)
    predicted_mask = create_mask(predicted_image)
    
    converted_mask = np.dot(mask,[0.299,0.587,0.114])

    m.update_state(predicted_mask,converted_mask)
  return m.result().numpy()

def test_one_image(image_test,path_weights):

    #Transform Image

    x = np.zeros((1,) + (256,256,3) , dtype="float32")
    x[0] = image_test

    #Predict Model
    model_avant_fit = model_compile()
    model = model_avant_fit.load_weights(path_weights)
    predicted_img = model.predict(x)

    #Create Mask
    format = 'png'
    epsilon = 0.595344
    max = np.max(predicted_img)
    eps = max*epsilon
    pred_mask = pred_mask > max-eps
    final_img = tf.keras.preprocessing.image.array_to_img(pred_mask)
    final_png = final_img.format
    return final_png