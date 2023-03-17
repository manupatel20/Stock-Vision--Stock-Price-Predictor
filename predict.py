import streamlit as st
import pickle as p
import numpy as np
import keras
import tensorflow as tf
import lstm

companies={
    "TCS",
    "Infosys",
    "Google",
    "Microsoft",

}

    

data= tf.keras.models.load_model('./saved_model.h5')
# regressor =data["model"]


def show_predict_page():
    st.title("Stock Price Prediction ")

    st.write("""### Predict prices""")
    company= st.selectbox("Company", companies)

    predict_but=st.button("Predict Prices")
    # st.write(predicted_val)
    if predict_but:
        # X=np.array([companies])
        price =data.predict(lstm.X_test)
        st.write(price)






