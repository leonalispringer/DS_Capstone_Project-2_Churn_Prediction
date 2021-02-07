import streamlit as st
import pickle
import pandas as pd

html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Predict Churn of Your Employee</h2>
</div><br><br>"""

from PIL import Image
img = Image.open("eliar.jpg")
st.image(img)

st.title('Welcome to the ELIAR LLC Employee Churn App')
st.subheader('In this project, Osman or any other employee will be able to predict the churn status, regarding the performance and features of her/him')

st.write("Guys, don't hesitate to try!")
         
st.markdown(html_temp,unsafe_allow_html=True)


selection=st.selectbox("Select Your Model", ["Gradient Boost", "KNN", "Random Forest"])

if selection =="Gradient Boost":
	st.write("You selected", selection, "model")
	model= pickle.load(open('gb_model', 'rb'))

elif selection =="KNN":
	st.write("You selected", selection, "model")
	model= pickle.load(open('knn_model', 'rb'))
    
else:
	st.write("You selected", selection, "model")
	model= pickle.load(open('rf_model', 'rb'))
    
satisfaction_level=st.slider("What is the satisfaction level of the employee?", 0.00,1.00, step=0.01)
st.write(satisfaction_level)
last_evaluation=st.slider("What is the last evaluation of the employee?", 0.00,1.00, step=0.01)
st.write(last_evaluation)
average_monthly=st.slider("What is the average monthly work hour of employee?", 0,350, step=1) 
st.write(average_monthly)
number_project=st.selectbox("What is the number of project that employee involved?",(0,1,2,3,4,5,6,7,8,9,10))
time_spend_company=st.selectbox("How many years has the employee spent so far?",(0,1,2,3,4,5,6,7,8,9,10))
    

my_dict = {'satisfaction_level': satisfaction_level,
 'last_evaluation': last_evaluation,
 'number_project': number_project,
 'average_montly_hours': average_monthly,
 'time_spend_company': time_spend_company}


columns=['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'salary', 'DeptType_1', 'DeptType_2',
       'DeptType_3', 'DeptType_4', 'DeptType_5', 'DeptType_6', 'DeptType_7',
       'DeptType_8', 'DeptType_9']

dfp = pd.DataFrame.from_dict([my_dict])

X = pd.get_dummies(dfp).reindex(columns=columns, fill_value=0)

prediction = model.predict(X)

print(prediction)

if prediction == 1:
    st.error("Your employee will LEAVE!")
else:
    st.success("Your employee will STAY!")           
    


     
   
    





    


