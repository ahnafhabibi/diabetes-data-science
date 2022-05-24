import streamlit as st
import joblib
import pandas as pd
st.write("# Diabetics Prediction")

gender = st.selectbox("Enter your gender",["Male", "Female"])
col1, col2, col3 = st.columns(3)

# getting user inputgender = col1.selectbox("Enter your gender",["Male", "Female"])

polydipsia=col3.selectbox("Do you drink a lot of water and still become thristy?",["Yes","No"])

age = col2.number_input("Enter your age")

polyuria = col1.selectbox("Do you excrete excessive urine?",["Yes","No"])

sudden_weight_loss= col1.selectbox("Have you experienced sudden weight loss?",["Yes","No"])

weakness= col2.selectbox("Do you feel physically weak?",["Yes","No"])

polyphagia=col3.selectbox("Do you consume food that exceeds your regular calorie count?",["Yes","No"])

genital_thrush= col1.selectbox("Do you have an infection in your private area?",["Yes","No"])

blurry_vision=col2.selectbox("Do you have blurry vision?",["Yes","No"])

itching=col3.selectbox("Do you have itching skin?",["Yes","No"])

irritability= col1.selectbox("Do you feel irritable quite easily?",["Yes","No"])

delayed_healing=col2.selectbox("Does your wound delay to heal?",["Yes","No"])

partial_paresis=col3.selectbox("Do you partial weakening of a muscle  or group of muscles?",["Yes","No"])

muscle_stiffness=col1.selectbox("Do you have face muscle stiffness after exercise or physical labour?",["Yes","No"])

alopecia=col2.selectbox("Do you have hairlosses often in clumps and shape of a quarter?",["Yes","No"])

obese=col3.selectbox("Are you obese?",["Yes","No"])


df_pred = pd.DataFrame([[age, gender, polyuria,polydipsia, sudden_weight_loss,
       weakness, polyphagia,genital_thrush,blurry_vision,
       itching, irritability,delayed_healing,partial_paresis,
       muscle_stiffness,alopecia, obese]],

columns= ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity'])

df_pred['Gender'] = df_pred['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

df_pred['Polyuria'] = df_pred['Polyuria'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['Polydipsia'] = df_pred['Polydipsia'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['sudden weight loss'] = df_pred['sudden weight loss'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['weakness'] = df_pred['weakness'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['Polyphagia'] = df_pred['Polyphagia'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['Genital thrush'] = df_pred['Genital thrush'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['visual blurring'] = df_pred['visual blurring'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['Itching'] = df_pred['Itching'].apply(lambda x: 1 if x == 'Yes' else 0)

df_pred['Irritability'] = df_pred['Irritability'].apply(lambda x: 1 if x == 'Yes' else 0)

df_pred['delayed healing'] = df_pred['delayed healing'].apply(lambda x: 1 if x == 'Yes' else 0)

df_pred['partial paresis'] = df_pred['partial paresis'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['muscle stiffness'] = df_pred['muscle stiffness'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['Alopecia'] = df_pred['Alopecia'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['Obesity'] = df_pred['Obesity'].apply(lambda x: 1 if x == 'Yes' else 0)

print(df_pred)
model = joblib.load('fhs_clf_model.pkl')
prediction = model.predict(df_pred)



if st.button("Predict"):
    if(prediction[0]==0):
        st.write('<p class="big-font">You are not diabetic.</p>',unsafe_allow_html=True)
    else:
        st.write('<p class="big-font">You are likely to be diabetic or have diabetics. Please consult a doctor as soon as possible</p>',unsafe_allow_html=True)
