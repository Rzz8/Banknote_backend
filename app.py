# 1. Import libraries
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

# 2. Create the app object
app = FastAPI()

origins = ['*']

app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=False,
                   allow_methods=["*"],
                   allow_headers=["*"])

pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

@app.get('/')
def main():
    return {"hello": "world"}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data: BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    print("hello")
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    if (prediction[0] > 0.5):
        prediction = "Fake note"
    else:
        prediction = "Its a Bank note"
    return {'prediction': prediction}


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn app:app --reload