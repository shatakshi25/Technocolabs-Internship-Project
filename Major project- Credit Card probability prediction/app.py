import streamlit as st
import pickle
import numpy as np
import pandas as pd

model =  pickle.load(open('model.pkl','rb'))

df = pd.read_csv('/content/drive/My Drive/Technocolabs project/cleaned_data.csv')
x = df.drop(['ID','SEX','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','default payment next month', 'EDUCATION_CAT', 'graduate school',
       'high school', 'others', 'university'],axis=1)

k = np.array(x.columns)

def prediction(features):

    features = np.array(features).astype(np.float64).reshape(1,-1)
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    return prediction, probability

def main():
    st.title('Prediction')
    html_temp="""
    <div style="background-color:orange; padding:15px;">
    <h2 style="text-align:center;font-family:verdana; font-size:300%; color:black;">PREDICTION BY INPUTS</h2>
    </div>
    <div >
    <p style="color:red;font-family:aerial">FOR STATUS 1 YOU HAVE DEFAULTED ACCOUNT AND FOR 0 YOU ARE SAFE</p>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    l=[]
    for i in range(0,17):
        l.append(st.text_input(k[i],"Type Here"))
        
    defaulted_html="""
    <div style="background-color:red;padding:12px;">
    <h3 style="color:white;text-align:center">You are not safe</h3>
    </div>
    """
    
    notdefaulted_html="""
    <div style="background-color:blue;padding:12px;">
    <h3 style="color:white;text-align:center">You are safe</h3>
    </div>
    """
    
    if st.button("Check"):
        output = prediction(l)
        st.success("The account status is :{}".format(output))
        
        if output==1:
            st.markdown(defaulted_html,unsafe_allow_html=True)
        else:
            st.markdown(notdefaulted_html, unsafe_allow_html=True)
            
if __name__=='__main__':
    main()