import pandas as pd
import ploty.express as px
import numpy as np
from Hackaton_Quantum.evaluation.predict import proba, evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def prompt_roc_curve(vgg_model,
                     inception_model,
                     silos,
                     pas_silos,
                     pas = 15):
  
  """Calculate the metrics depending on the threshold. Calculate the AUC of ROC """

  df_results = pd.DataFrame(columns = ['confusion_matrix','precision','recall','F1-score','TPR','FNR','TNR','FPR','k','model'])
  
  vgg_probas_silos, vgg_probas_pas_silos = proba(vgg_model, silos, pas_silos)
  
  inc_probas_silos, inc_probas_pas_silos = proba(inception_model, silos, pas_silos)

  vgg_auc = roc_auc_score(np.array([[score_silos, 1-score_silos] for scores_silos in vgg_probas_silos] 
                          + [[score_pas_silos, 1-score_pas_silos] for score_pas_silos in vgg_probas_pas_silos]))
  inc_auc = roc_auc_score(np.array([[score_silos, 1-score_silos] for scores_silos in inc_probas_silos] + [[score_pas_silos, 1-score_pas_silos] for score_pas_silos in inc_probas_pas_silos]))
  
  for k in range(1,pas):
    _threshold = (float(k)) / pas
    dict_result = evaluate(vgg_probas_silos, vgg_probas_pas_silos, threshold =_threshold)
    dict_result['k'] = k
    dict_result['model'] ='vgg_model'
    dict_result['auc'] = vgg_auc
    df_results.loc[2*(k-1)] = list(dict_result.values())
    
    dict_result = evaluate(inc_probas_silos, inc_probas_pas_silos, threshold = _threshold)
    dict_result['k'] = k
    dict_result['model'] ='inception_model'
    dict_result['auc'] = inc_auc
    df_results.loc[2*(k-1)+1] = list(dict_result.values())

    #px.line(df_results, x = 'FPR', y = 'TPR',color = 'model').show()
    #px.line(df_results, x = 'recall', y = 'precision',color = 'model').show()
    #px.line(df_results, x = 'k',y = 'F1-score',color= 'model').show()
  return df_results


def display(display_list):
  """ Call shows_predictions to display images"""
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()-
def show_predictions(dataset, num=6):
  """
  Displays the first image of each of the num batches
  """
  if dataset:
    for image,mask in dataset:
      model_compiled = model_compile()
      model_compiled.load_weights('weights100.hdf5')
      pred_mask = model_compiled.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)[0]])
      #display([image[0], mask[0], pred_mask[0]])

  else:
    display([sample_image, sample_mask,
    create_mask(model_compiled.predict(sample_image[tf.newaxis, ...]))])