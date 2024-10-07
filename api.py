from fastapi import FastAPI, File, UploadFile,Request,HTTPException
import tensorflow as tf
import json
from model_definition import SegmentationModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import cv2 
import numpy as np
import io
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate
from typing import List
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from starlette.responses import StreamingResponse
from pathlib import Path


app = FastAPI()

# templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")
# # app.mount("/processed_images", StaticFiles(directory="processed_images"), name="processed_images")


model = SegmentationModel().model
model.load_weights('model.h5')



# @app.get("/", response_class=HTMLResponse)
# async def read_index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-to-image/")
async def predict_to_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        np_array = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image file")

        preprocessed_image = SegmentationModel.preprocess_image(image, (256, 256))
        predictions = model.predict(preprocessed_image)
        
        # Convert predictions to an image
        predicted_image = predictions[0, :, :, 0]  # Assuming single-channel output
        predicted_image = (predicted_image * 255).astype(np.uint8)  # Rescale to [0, 255]

        # Remove hair using the mask
        hair_free_image = SegmentationModel.remove_hair_with_mask(image, predicted_image)

        # Encode image to bytes
        is_success, buffer = cv2.imencode(".png", hair_free_image)
        if not is_success:
            raise ValueError("Failed to encode image")
        
        io_buf = io.BytesIO(buffer)
        return StreamingResponse(io_buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/remove_hair")
# async def scoring_endpoint(file: UploadFile = File(...)):
#     try:
#         file_bytes = await file.read()
#         np_array = np.frombuffer(file_bytes, np.uint8)
#         image = cv.imdecode(np_array, cv.IMREAD_COLOR)

#         if image is None:
#             raise ValueError("Invalid image file")

#         preprocessed_image = SegmentationModel.preprocess_image(image, (256, 256))
#         mask = model.model.predict(preprocessed_image)[0]

#         # Convert the mask to the same size as the input image
#         mask = cv.resize(mask, (image.shape[1], image.shape[0]))
#         mask = (mask * 255).astype(np.uint8)  # Convert mask to uint8

#         # Remove hair
#         hair_free_image = SegmentationModel.remove_hair_with_mask(image, mask)

#         # Save the result to a file
#         result_path = Path("processed_images") / file.filename
#         cv.imwrite(str(result_path), hair_free_image)

#         return JSONResponse(content={"result_path": f"/processed_images/{file.filename}"})

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))