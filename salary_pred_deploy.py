# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:25:53 2023

@author: devid
"""

import pickle as pkl
import numpy as np
import streamlit as st
st.title("SALARY PREDICTION")

filepath ="C:\\Users\\devid\\sal_model.sav"
model=pkl.load(open(filepath,"rb"))

def pred(x):
    x=np.array(x).reshape(1,-1)
    result=model.predict(x)
    return(result)
         
def main():
    yearofexperience=st.number_input("Years Of Experience")
    new_data=[yearofexperience]
    if st.button("Predict"):
        st.write(pred(new_data))

if __name__=="__main__":
    main()
    