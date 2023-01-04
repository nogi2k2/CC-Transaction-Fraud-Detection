import uvicorn
import joblib
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, File, Query, UploadFile,HTTPException, Form
from fastapi.responses import FileResponse, PlainTextResponse

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="""An API that utilises a Machine Learning model that detects if a credit card transaction is fraudulent or not.""",
    version="1.0.0", debug=True)

# model = joblib.load('credit_card_fraud.pkl')
favicon = 'favicon.png'

@app.get("/", response_class=PlainTextResponse)
async def running():
    note="""
    Credit Card Fraud Detection API is Running"""
    return note

@app.get('/favicon.png', include_in_schema=False)
async def favicon():
    return FileResponse(favicon)

class fraudDetection(BaseModel):
    step:int
    types:int
    amount:float	
    oldbalanceorig:float	
    newbalanceorig:float	
    oldbalancedest:float	
    newbalancedest:float	
    isflaggedfraud:float

@app.post('/predict')
def predict(data: fraudDetection):
    model = joblib.load("credit_card_fraud.pkl")
    features = np.array([[data.step, data.types, data.amount, data.oldbalanceorig, data.newbalanceorig, data.oldbalancedest, data.newbalancedest, data.isflaggedfraud]])
    prediction = model.predict(features)
    if prediction==1:
        return {"Fraudulent"}
    elif prediction==0:
        return {"Not Fraudulent"}
    return 
