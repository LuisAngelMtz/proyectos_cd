import pandas as pd
import numpy as np

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

def freq(df, var):
    if type(var) != list:
        var = [var]
    for v in var:
        aux = df[v].value_counts().to_frame().rename(columns={v:'FA'})
        aux['FR'] = aux['FA'] / aux['FA'].sum()
        aux[['FAA','FRA']] = aux.apply( np.cumsum )
        print(f'Tabla de frecuencias para la variable {v} \n')
        print(aux,'\n') 

              
def norm(df, v, umbral):
    aux = df[v].value_counts(True).to_frame()
    aux[f'n_{v}'] = np.where( aux[v] < umbral , 'CAT PEQUEÑAS', aux.index )
    valor = aux.head(1)[f'n_{v}'].values[0]
    if aux.loc[aux[f'n_{v}'] == 'CAT_PEQUEÑAS'][v].sum() < umbral:
        aux[f'n_{v}'].replace( {'CAT_PEQUEÑAS':valor} , inplace=True )
               
    aux.drop(v,axis=1, inplace=True)
    aux.reset_index(inplace=True)
               
    return df.merge(aux, left_on=[v], right_on = ['index'], how='left').drop('index',axis=1)   
    
def discretizar(df, v, k):

    kb = KBinsDiscretizer( n_bins=k , encode='ordinal', strategy='quantile' )    
    kb.fit(df[[v]])
    df[f'd_{v}_{k}'] = pd.cut( df[v], bins= kb.bin_edges_[0] , include_lowest=True  ).astype(str)
    
    return df

def calculo_iv(df, v, tgt , um):
    aux = df.pivot_table( index = v , 
                          columns = tgt , 
                          values = um , 
                         aggfunc = 'count', 
                         fill_value = 0 )
    
    aux[list(range(2))] = aux/aux.apply(np.sum)
    aux['w'] = np.log( aux[0] / aux[1] )
    aux['iv'] = ( aux[0] - aux[1] ) * aux['w']
    
    return v, aux['iv'].sum()
    
def codificacion_woe(df,v,tgt,um):
    
    aux = df.pivot_table(index=v,
                         columns=tgt,
                         values=um,
                         aggfunc='count',
                         fill_value=0)
    
    aux[list(range(2))] = aux/aux.apply(np.sum)
    
    aux['w'] = np.log(aux[0]/aux[1])
    
    aux.drop(range(2),axis=1,inplace=True)
    
    aux = aux.to_dict()['w']
    
    return v,aux

def metricas(model,Xv,yv):
    print(" Métricas para modelo de clasificación: \n")

    print(" Valor ROC : %.3f"   %roc_auc_score( y_score=model.predict_proba(Xv)[:,1] , y_true=yv  )   )

    print(" Valor ACC : %.3f\n" %accuracy_score( y_pred=model.predict(Xv) , y_true=yv) )

    print(" Matriz de confusión: ", "\n", confusion_matrix(y_pred=model.predict(Xv) , y_true=yv ) )

