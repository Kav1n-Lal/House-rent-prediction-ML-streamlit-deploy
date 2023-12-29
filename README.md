# House-rent-prediction-ML-streamlit-deploy
- App is deployed on this url-->>[https://house-rent-prediction-ml-app-deploy-yiyotjs46xxm4phxwxtdwn.streamlit.app/]
###  Project Demonstration video links:
- **Part1(Intro_ipynb_explanation)**
- [https://drive.google.com/file/d/1vEJiSH4jW521hPM6zbgx8YdHHo5vWPkb/view?usp=sharing]
- **Part2(EDA)**
- [https://drive.google.com/file/d/1YciZt2I7PMlQqS5oNs35911NoYKf47zj/view?usp=sharing]
- **Part3(pickling_deployment)**
- [https://drive.google.com/file/d/1CFtv5bRy2p01JI01WrJj0B5l98UXOuwt/view?usp=sharing]
- View **House_rent_EDA.ipynb** to view various insights that could be obtained through EDA on the dataset.
- View **house_rent_prediction.ipynb** -where I have done a little bit of EDA to gain insights, used SHAP analysis to select the best features and trained various models.
- View **Final_rent_price.ipynb** -the refined notebook where I have used only the *selected features* obtained through SHAP and used the **HistGradientBoostingRegressor** model to run the entire dataset and save the model as a pickle file named **houserent_saved_steps_regressor.pkl**.    
- - ## Regression R2 and RMSE score Table
|    Model             |  Train(R2-score)   |  Train(RMSE)      | Test(R2-score)     |  Test(RMSE)       |
| :------------------- | -----------------  |-----------------: | -----------------  |-----------------: |
| Linear Regression    |      0.928         |0.057              | 0.923              |0.058              |
| RandomForest         |      0.99          |0.023              | 0.927              |0.057              |
|HistGradientBoostingRegressor|0.962        |0.041              | 0.928              |0.056              |
- I used the **app.py** to create a **Streamlit** app and deployed live using **Streamlit Cloud**.-->>[https://house-rent-prediction-ml-app-deploy-yiyotjs46xxm4phxwxtdwn.streamlit.app/]
