import streamlit as st

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import plotly.express as px

def generate_house_data(n_samples=100): 
    np.random.seed(50)
    size = np.random.normal(1400,50 , n_samples) #normal gaussian distribution

    price = size * 50 +  np.random.normal(0,50,n_samples)
    
    return pd.DataFrame({'size': size, 'price': price})
    

  ###train the model

def train_model():
   df = generate_house_data(n_samples=100)
   X=df[['size']]
   y=df[['price']]

   X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

   model = LinearRegression()
   model.fit(X_train, y_train)
   return model
 

def main():

    st.title("Simple Linear Regression House Prediction App")
    st.write("enter in house size to know its price ")


    model = train_model()

    size =st.number_input('House size',min_value=500,max_value=2000,value=1500)

    if st.button('Predict Price'):

        input_data = pd.DataFrame([[size]],)
        predicted_price = model.predict(input_data)
      
        st.success(f'Estimated price: ${float(predicted_price[0].item()):.2f}')
 
        df =generate_house_data()

        fig = px.scatter(df,x='size',y='price',title="house prediction size vs price")
        fig.add_scatter(x=[size] ,y=[predicted_price],
                          mode='markers',
                          marker=dict(size=15,color='red', line_width=2),
                          name='Prediction',
                          cliponaxis=False
                          )
                            

        st.plotly_chart(fig)



if __name__ == "__main__":
    main()