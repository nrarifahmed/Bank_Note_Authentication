# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 01:36:08 2020

@author: Arif Ahmed NR nrarifahmed@gmail.com
"""
#pip install streamlit

import numpy as np
import pandas as pd
import pickle
import streamlit as st

from PIL import Image

pickle_in=open("bank_note_classifier.pkl","rb")
bank_note_classifier=pickle.load(pickle_in)

def Welcome():
    return "Welcome All"

def predict_bank_note_auth(variance,skewness,curtosis,entropy) :
    """ Logistic Regression Function BANK NOTE AUTHENTICATION PREDICTION
    This Is Using docstrings For Specification.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
        responses:
            200:
                description: The Output Values                       
    """
    
    prediction=bank_note_classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction
               
def main():
    st.title("Bank Note Authenticator")
    html_temp="""
    <div style"background-color:blue',padding:10px">
    <h2 style="color:white:text-align:center;"> Created By : Arif Ahmed nrarifahmed@gmail.com </h2>
    <h2 style="color:white:text-align:center;"> Streamlit Bank Authentication ML App </h2>
    </div>
    
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    variance=st.text_input("variance","Type Here")
    skewness=st.text_input("skewness","Type Here")
    curtosis=st.text_input("curtosis","Type Here")
    entropy=st.text_input("entropy","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_bank_note_auth(variance,skewness,curtosis,entropy)
        st.success('The Output Is {}'.format(result))
    if st.button("About"):
        st.text("lets's Learn")
        st.text("Built With Streamlit API")
    

if __name__ == '__main__':
    main()






