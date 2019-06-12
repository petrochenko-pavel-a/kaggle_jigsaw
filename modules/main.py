from musket_core import preprocessing,datasets,model,tasks
from musket_core.datasets import PredictionItem
from musket_core.context import get_current_project_path
import pandas as pd
import numpy as np
import benchmark

@preprocessing.dataset_preprocessor
def binarize_target(inp:PredictionItem):
    inp.y=inp.y>0.5
    return inp

class PDDataSet(datasets.DataSet):
    
    def __init__(self,df,targetColumn:str,featureColumn,idColumn:str=None,sep=","): 
        self.df=df
        self.feature=self.df[featureColumn].values
        
        if (targetColumn in self.df.columns):
            self._target=self.df[targetColumn].values
        else:
            self._target=np.zeros((len(self),1))
        if idColumn is not None:    
            self.ids=self.df[idColumn].values        
        else:
            self.ids=list(range(len(self.df)))   
        pass    
    
    def __len__(self):
        return round(len(self.df))
        
    def __getitem__(self, item)->datasets.PredictionItem:
        return PredictionItem(self.ids[item],self.feature[item],np.array([self._target[item]]))
    
    def get_target(self,item):
        return np.array([self._target[item]])

_ds=None
_data=None

@datasets.dataset_provider
def get_train():
    global _data
    global _ds
    if _data is None:
        _data=pd.read_csv(get_current_project_path()+"/data/train.csv")
        _ds=PDDataSet(_data, "target","comment_text","id")
    return _ds    

@tasks.task        
def measure(m:model.ConnectedModel,d:datasets.DataSet):    
    newD=pd.DataFrame(_data.iloc[d.indexes])    
    preds=m.predictions(d)
    newD["model"]=pd.Series(np.array(preds.predictions).reshape(-1),index=newD.index)
    return benchmark.validate(newD)   