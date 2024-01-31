import streamlit as st
import pickle as p
import numpy as np
import keras
import tensorflow as tf
import lstm
import matplotlib.pyplot as plt
from lstm import test_set
import xTest
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# import new_final

companies=[
    "AAPL",
    "BAC",
    "MDB",
    "TSLA",
    "UBER"

]



    

   

data= tf.keras.models.load_model('./final_model_h5.h5')
# regressor =data["model"]

def plot(test,predicted):
    st.pyplot(test, color='red',label='Real Apple Stock Price')
    st.pyplot(predicted, color='blue',label='Predicted Apple Stock Price')
    st.pyplot.title('Apple Stock Price Prediction')
    st.pyplot.xlabel('Time')
    st.pyplot.ylabel('Apple Stock Price')
    st.pyplot.legend()
    st.pyplot.show()


def show_predict_page():
    st.title("Stock Price Prediction ")

    st.write("""### Predict prices""")
    company= st.selectbox("Company", companies)

    
    

    
    predict_but=st.button("Predict Prices")
    # st.write(predicted_val)
    if predict_but:
        for i in companies:
         if(i==company):
            company_index=companies.index(i)
            # xTest.test(company_index)
        # X=np.array([companies])

        FEATURES=['Close', 'Company']
        sequence_length=50
        dataset = pd.read_csv('./dataset_new.csv')
        df_temp = dataset.loc[dataset['Company'] == company_index]

        X_test= df_temp[-sequence_length:]
        X_test = X_test.filter(FEATURES)
        print(X_test)


        data_filtered = dataset[FEATURES]
        data_filtered_ext = data_filtered.copy()
        data_filtered_ext['Prediction'] = data_filtered_ext['Close']

        scaler_pred = MinMaxScaler()
        scaler_pred = MinMaxScaler()
        df_Close = pd.DataFrame(data_filtered_ext['Close'])
        np_Close_scaled = scaler_pred.fit_transform(df_Close)
        price =data.predict(np.array(xTest.test(company_index)))
        pred_price_unscaled = scaler_pred.inverse_transform(price.reshape(-1, 1))

        st.write(pred_price_unscaled)
        # plot(test_set,price)
        # st.line_chart(price)
        st.line_chart(np.array(X_test))






