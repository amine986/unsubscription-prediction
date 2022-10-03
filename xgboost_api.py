
    import pandas as pd
    from pycaret.MLUsecase.CLASSIFICATION import load_model, predict_model
    from fastapi import FastAPI
    import uvicorn
    # Create the app
    app = FastAPI()
    # Load trained Pipeline
    model = load_model('xgboost_api')
    # Define predict function
    @app.post('/predict')
    def predict(CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_France, Geography_Germany, Geography_Spain, Gender_Female, Gender_Male):
        data = pd.DataFrame([[CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_France, Geography_Germany, Geography_Spain, Gender_Female, Gender_Male]])
        data.columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_France', 'Geography_Germany', 'Geography_Spain', 'Gender_Female', 'Gender_Male']
        predictions = predict_model(model, data=data)
        return {'prediction': list(predictions['Label'])}
    if __name__ == '__main__':
        uvicorn.run(app, host='127.0.0.1', port=8000)