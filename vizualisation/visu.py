import pandas as pd
import ploty.express as px
import numpy as np
from Hackaton_Quantum.evaluation.predict import proba, evaluate

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