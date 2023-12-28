import streamlit as st  
import pandas as pd
import pickle


from sklearn.model_selection import train_test_split
# for encoding with feature-engine
from feature_engine.encoding import MeanEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error

sheet_id = '1TWNez8kF5LpTc0QF1UuYrdmLXxJhGV2mVTpjGU7Sbt0'
sheet_name = 'House_Rent_Prediction.csv'
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
df = pd.read_csv(url)

def main():
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> House Rent Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.write("""### We need some information to predict the rent""")
    
    def load_model():
            with open('houserent_saved_steps_regressor.pkl','rb') as file:
                data=pickle.load(file)
            return data

    data=load_model()

    rent_box_cox_transform=data['transformer1_rent']
    prop_size_yeo_johnson_transform=data['transformer2_prop_size']
    mean_enc=data['mean_enc']
    scaler=data['scaler']
    regressor_loaded=data['model']

    z=pd.read_csv('rent_features.csv')

    # Storing all user input 
    input_dict={'type':'', 'locality':'', 'gym':'', 'swimming_pool':'', 'negotiable':'',
            'parking':'', 'property_size':'', 'property_age':'', 'bathroom':'',
            'cup_board':'', 'floor':'', 'total_floor':'',  'INTERNET':'', 'AC':'', 'CLUB':'',
            'CPA':'', 'SECURITY':'', 'GP':'', 'PARK':'','RWH':'', 'STP':'','rent_level':''}

    form = st.form("form1")



    type=form.selectbox('Choose housetype',list(z['type'].unique()))
    input_dict['type']=type

    locality=form.selectbox('Choose locality',list(z['locality'].unique()))
    input_dict['locality']=locality.split(',')[0]
    #print(input_dict['locality'])

    gym=form.selectbox('Want gym',['Yes',"No"])
    if gym=='Yes':
        input_dict['gym']=1
    else:
        input_dict['gym']=0
        
    swimming_pool=form.selectbox('Want swimming_pool',['Yes',"No"])
    if swimming_pool=='Yes':
        input_dict['swimming_pool']=1
    else:
        input_dict['swimming_pool']=0

    negotiable=form.selectbox('Want to negotiate rent price',['Yes',"No"])
    if negotiable=='Yes':
        input_dict['negotiable']=1
    else:
        input_dict['negotiable']=0
        
    parking=form.selectbox('What vehicle do you need to park',list(z['parking'].unique()))
    input_dict['parking']=parking



    property_size=form.selectbox('What is your expected property_size',list(z['property_size'].unique()))
    input_dict['property_size']=property_size

    property_age=form.selectbox('What is your expected property_age',list(z['property_age'].unique()))
    input_dict['property_age']=property_age

    bathroom=form.selectbox('How many bathrooms do you need',list(z['bathroom'].unique()))
    input_dict['bathroom']=bathroom

    cup_board=form.selectbox('How many cupboards do you need',list(z['cup_board'].unique()))
    input_dict['cup_board']=cup_board

    floor=form.selectbox('Preferred floor size',list(z['floor'].unique()))
    input_dict['floor']=floor

    total_floor=form.selectbox('How many floors do you need',list(z['total_floor'].unique()))
    input_dict['total_floor']=total_floor


            
    INTERNET=form.selectbox('Want internet connection',['Yes',"No"])
    if INTERNET=='Yes':
        input_dict['INTERNET']=1
    else:
        input_dict['INTERNET']=0
        
    AC=form.selectbox('Want AC',['Yes',"No"])
    if AC=='Yes':
        input_dict['AC']=1
    else:
        input_dict['AC']=0
        
    CLUB=form.selectbox('Want CLUB',['Yes',"No"])
    if CLUB=='Yes':
        input_dict['CLUB']=1
    else:
        input_dict['CLUB']=0
        
    CPA=form.selectbox('Want CPA',['Yes',"No"])
    if CLUB=='Yes':
        input_dict['CPA']=1
    else:
        input_dict['CPA']=0

    SECURITY=form.selectbox('Want SECURITY',['Yes',"No"])
    if SECURITY=='Yes':
        input_dict['SECURITY']=1
    else:
        input_dict['SECURITY']=0
        
    GP=form.selectbox('Want GP',['Yes',"No"])
    if GP=='Yes':
        input_dict['GP']=1
    else:
        input_dict['GP']=0
        
    PARK=form.selectbox('Want Park',['Yes',"No"])
    if PARK=='Yes':
        input_dict['PARK']=1
    else:
        input_dict['PARK']=0
        
    RWH=form.selectbox('Want RWH',['Yes',"No"])
    if RWH=='Yes':
        input_dict['RWH']=1
    else:
        input_dict['RWH']=0
        
    STP=form.selectbox('Want STP',['Yes',"No"])
    if STP=='Yes':
        input_dict['STP']=1
    else:
        input_dict['STP']=0
        
    rent_level=form.selectbox('Select rent level',list(z['rent_level'].unique()))
    input_dict['rent_level']=rent_level

    ok=form.form_submit_button("PREDICT") # 
    safe_html ="""  
        <div style="background-color:#80ff80; padding:10px >
        <h2 style="color:white;text-align:center;"> Result</h2>
        </div>
        """  
    if ok:
        #st.write(input_dict)
        A=pd.DataFrame(input_dict,index=[1])
        
        A['property_size']= prop_size_yeo_johnson_transform.transform(A[['property_size']])
        A=mean_enc.transform(A)
        A=scaler.transform(A)
        #st.write(A)
        y_pred=regressor_loaded.predict(A.reshape(1,-1))
        t=rent_box_cox_transform.inverse_transform(y_pred.reshape(-1,1))
        u=str(round((list(t)[0][0]),2))+'/-'
        st.success(f'### Predicted Rent Price is : {u}')
        if t[0][0]>10000:
            st.markdown(safe_html,unsafe_allow_html=True)
        

if __name__=='__main__':
        main()