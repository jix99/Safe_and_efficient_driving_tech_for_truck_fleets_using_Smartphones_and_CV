#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
import streamlit as st
from PIL import Image

classes = {'c0': 'normal driving',
    'c1': 'texting - right',
    'c2': 'talking on the phone - right',
    'c3': 'texting - left',
    'c4': 'talking on the phone - left',
    'c5': 'operating the radio',
    'c6': 'drinking',
    'c7': 'reaching behind',
    'c8': 'hair and makeup',
    'c9': 'talking to passenger',}



# In[30]:

st.markdown("<h1 style='text-align: center; color: red;'>Safe and efficient driving tech for truck-fleets using Smartphones and CV \n</h1>", unsafe_allow_html=True)
uploaded_data = st.file_uploader('test_data')


if uploaded_data is not None:
    st.subheader('Input:\n')
    img = Image.open(uploaded_data)
    st.image(uploaded_data, caption = 'Input Image')
    img = cv2.resize(cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB) ,(64,64))
    


    # In[31]:



    img = img.astype('float32') / 255


    # In[32]:


   


    # In[20]:


    model = tf.keras.models.load_model('saved_model.pb')


    # In[33]:


    img = img.reshape(-1, 64, 64, 3)
    


    # In[28]:


    

    # In[34]:


    model.predict(img)


    # In[35]:

    st.subheader('Output:\n')
    st.write(classes['c'+ str(np.argmax(model.predict(img)))])


    # In[ ]:




