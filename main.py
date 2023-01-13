from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import time
from ipynb.fs.full.Inference import predict
app=FastAPI(
    docs_url="/docs",
    title="Crowd Density Model API",
    description="This API is counting number of people in the crowd",
    version="1.0.00"
    
    
)
class BasicResponse(BaseModel):
    status_code : int
    message : str
    content : Optional[dict]
@app.get("/",response_model=BasicResponse)

def root():
    
#     -**sample** : {
#         "status_code": 200,
#         "message": "healthy",
#         "content":{"timestamp": "2023-01-12T11:20:51.607672"}
#     }
#     -**type**: json
    return BasicResponse(status_code=200,message="healthy",content={
        "timestamp": datetime.now()
        
    })
class DetectionInput(BaseModel):
    image:str
    class Config:
        schema_extra={
            "example":{"image":"C:/Users/User/Crowd-Density/dataset/part_B_final/test_data/images/IMG_20.jpg"
                      }
        }
@app.post("/detect",response_model=BasicResponse)

def detection(inp:DetectionInput):
    time_start=time.time()
    try:
        count = predict(inp.image)
        print("after predict")
    except Exception as e:
        return BasicResponse(status_code=400,message=str(e),content={
        "timestamp": datetime.now()
        
    })
    time_end=round(time.time()- time_start,2)
    return BasicResponse(status_code=200,message="success",content={
       "number_of_people":count,
        "computation_time":f"{time_end} seconds",
        "timestamp": datetime.now()
        
    })
        