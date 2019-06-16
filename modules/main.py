from musket_core import preprocessing,datasets,model,tasks
from musket_core.datasets import PredictionItem
from musket_core.context import get_current_project_path
import pandas as pd
import numpy as np
import benchmark
import keras.backend as K
import keras.utils as utils

def expand_multiple_dims(x, axes, name="expand_multiple_dims"):
  """
  :param tf.Tensor x:
  :param list[int]|tuple[int] axes: after completion, tf.shape(y)[axis] == 1 for axis in axes
  :param str name: scope name
  :return: y where we have a new broadcast axis for each axis in axes
  :rtype: tf.Tensor
  """
  with tf.name_scope(name):
    for i in sorted(axes):
      x = tf.expand_dims(x, axis=i, name="expand_axis_%i" % i)
    return x


def dimshuffle(x, axes, name="dimshuffle"):
  """
  Like Theanos dimshuffle.
  Combines tf.transpose, tf.expand_dims and tf.squeeze.

  :param tf.Tensor x:
  :param list[int|str]|tuple[int|str] axes:
  :param str name: scope name
  :rtype: tf.Tensor
  """
  with tf.name_scope(name):
    assert all([i == "x" or isinstance(i, int) for i in axes])
    real_axes = [i for i in axes if isinstance(i, int)]
    bc_axes = [i for (i, j) in enumerate(axes) if j == "x"]
    if x.get_shape().ndims is None:
      x_shape = tf.shape(x)
      x = tf.reshape(x, [x_shape[i] for i in range(max(real_axes) + 1)])  # will have static ndims
    assert x.get_shape().ndims is not None

    # First squeeze missing axes.
    i = 0
    while i < x.get_shape().ndims:
      if i not in real_axes:
        x = tf.squeeze(x, axis=i)
        real_axes = [(j if (j < i) else (j - 1)) for j in real_axes]
      else:
        i += 1

    # Now permute.
    assert list(sorted(real_axes)) == list(range(x.get_shape().ndims))
    if real_axes != list(range(x.get_shape().ndims)):
      x = tf.transpose(x, real_axes)

    # Now add broadcast dimensions.
    if bc_axes:
      x = expand_multiple_dims(x, bc_axes)
    assert len(axes) == x.get_shape().ndims
    return x

@preprocessing.dataset_preprocessor
def binarize_target(inp:PredictionItem):
    inp.y=inp.y>0.5
    return inp

def soft_AUC(y_true, y_pred):
    # Extract 1s
    pos_pred_vr = tf.boolean_mask(y_pred,K.equal(y_true, 1))
        # Extract zeroes
    neg_pred_vr = tf.boolean_mask(y_pred,K.equal(y_true, 0))
    # Broadcast the subtraction to give a matrix of differences  between pairs of observations.
    pred_diffs_vr = dimshuffle(pos_pred_vr,[0, 'x']) - dimshuffle(neg_pred_vr,['x', 0])
    # Get signmoid of each pair.
    stats = K.sigmoid(pred_diffs_vr * 2)
    # Take average and reverse sign
    return 1-K.mean(stats) # as we want to minimise, and get this to zero

utils.get_custom_objects()["soft_auc"]=soft_AUC

import tensorflow as tf
from keras import backend as K

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

utils.get_custom_objects()["auc"]=auc



class PDDataSet(datasets.DataSet):
    
    def __init__(self,df,targetColumn:str,featureColumn,idColumn:str=None,sep=","): 
        self.df=df#[df["jewish"]>0.5]
        
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
    result=benchmark.validate(newD)
    print("result",result)
    return result   

@tasks.task
def all_blend(m:model.ConnectedModel):
    import test
    test.test_blend(m)
    pass


