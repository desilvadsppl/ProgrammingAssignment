from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Load the trained model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Create a FastAPI instance
app = FastAPI()

# Define the input data model
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the prediction endpoint
@app.post("/predict/")
def predict(data: IrisData):
    try:
        # Prepare the input data
        input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])

        # Make prediction
        prediction = model.predict(input_data)
        predicted_class = iris.target_names[prediction[0]]

        return {"prediction": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
