import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

st.set_page_config(
    page_title="Naive Bayes Algorithm"
)

st.write("""
# Dynamic _Iris flower_ prediction

Based on the user's input, this web app uses a **Naive Bayes Classifier** to predict the Iris flower variety.

""")

def user_input_features():
    sepal_length = st.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length' : sepal_length,
            'sepal_width' : sepal_width,
            'petal_length' : petal_length,
            'petal_width' : petal_width}

    features = pd.DataFrame(data, index = [0])
    return features

df = user_input_features()

st.subheader('User input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = GaussianNB()
clf.fit(X, Y)

prediction = clf.predict(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])