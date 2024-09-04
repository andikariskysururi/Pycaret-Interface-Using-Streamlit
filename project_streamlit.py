import streamlit as st  
import pandas as pd
import pycaret
import io 
from sklearn.model_selection import train_test_split
import pickle

def splitting_data(df):
    train, test = train_test_split(df, test_size = 0.2)
    return train, test  

def setup_pycaret(train, y, case):
    s = setup(train, target = y, session_id=123) 
    return s 

st.title("PYCARET INTERFACE USING STREAMLIT")
file_upload = st.file_uploader("File Upload CSV Format")
if file_upload is not None :
    
    filename = file_upload.name 
    filename = filename.split(".")[0]
    st.subheader(f"Dataframe {filename}")
    df = pd.read_csv(file_upload)
    st.dataframe(df)
    
    
    st.subheader("META DATA")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    # Display in Streamlit
    st.text(info_str)
    
    st.subheader("Select Data Target")
    column = df.columns
    case = st.selectbox('Pease Select Your Supervised Type', ['Classification', 'Regression'], index= None)
    if case== "Classification":
        from pycaret.classification import * 
    else : 
        from pycaret.regression import *
    y = st.selectbox('Pease Select Your Target Fitur', column, index= None)
    if y is not None : 
        st.write(f"Your data target is {y}")
        train, test = splitting_data(df)
        setup_pycaret(train,y, case)
        with st.spinner('Comparing models, please wait...'):
            # Long-running task
            best = compare_models()
        result = pull()
        st.dataframe(result)
        st.text(f"Your best model is {best}")
        
       
        
        
        
        st.subheader("FEATURE IMPORTANCE")
        plt_bound = plot_model(best,display_format="streamlit", save=True, plot = 'feature')
        st.image(plt_bound)
        
        if case == 'Classification': # Generate the plot
            st.subheader("BOUNDARY VISUALIZATION")
            plt_bound = plot_model(best,display_format="streamlit", save=True, plot = 'boundary')
            st.image(plt_bound)
            
            st.subheader("CONFUSSION MATRIX")
            plt_conf = plot_model(best, display_format="streamlit", plot='confusion_matrix', save=True)
            st.image(plt_conf)
            
            
            st.subheader("CLASS REPORT")
            plt_bound = plot_model(best,display_format="streamlit", save=True, plot = 'class_report')
            st.image(plt_bound)
            
        isSave = st.selectbox("Do you wanna Extract Your Model?", ['Yes', 'No'], index=None)    
        if isSave == 'Yes':
            # Save the model to a BytesIO object
            buf = io.BytesIO()
            pickle.dump(best, buf)
            buf.seek(0)
            
            # Provide a download button
            st.download_button(
                label="Download Model",
                data=buf,
                file_name="BestModel.pkl",
                mime="application/octet-stream"
            )
            st.success("Model is ready for download!")
    else : 
        st.write(f"Your data target is {y}")
else : 
    st.write("Please Upload Your Dataset")
    
    
    