import pandas as pd
import ploty.express as px
from Hackaton_Quantum.evaluation.predict import evaluate

def prompt_roc_curve(vgg_model,
                     inception_model,
                     silos,
                     pas_silos,
                     pas = 15):
  df_results = pd.DataFrame(columns = ['confusion_matrix','precision','recall','F1-score','TPR','FNR','TNR','FPR','k','model'])
  for k in range(1,pas):
    _threshold = (float(k) ) / pas
    dict_result = evaluate(vgg_model,silos,pas_silos,threshold = _threshold)
    dict_result['k'] = k
    dict_result['model'] ='vgg_model'
    df_results.loc[2*(k-1)] = list(dict_result.values())
    dict_result = evaluate(inception_model,silos,pas_silos,threshold = _threshold)
    dict_result['k'] = k
    dict_result['model'] ='inception_model'
    df_results.loc[2*(k-1)+1] = list(dict_result.values())
    px.line(df_results, x = 'FPR', y = 'TPR',color = 'model').show()
    px.line(df_results, x = 'recall', y = 'precision',color = 'model').show()
    px.line(df_results, x = 'k',y = 'F1-score',color= 'model').show()
  return df_results