import io

import numpy as np

import pickle

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image

from typing import List


classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


with open('static/model/model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.get("/", response_class=HTMLResponse)
@app.post("/", response_class=HTMLResponse)
def main():
    files = [file + '.jpg' for file in classes]
    paths = ['static/images/' + x for x in files]
    empty_column_labels = []
    content = html_open_tag + head_tag + body_center_open_tag + \
              marquee_home_tag + heading + table_tag(paths, classes, empty_column_labels) + \
              form_tag + body_center_close_tag + html_close_tag

    return content


@app.post("/predictions/", response_class=HTMLResponse)
async def render_predictions(files: List[UploadFile] = File(...)):
    pillow_images = []
    predictions = []
    image_paths = []

    new_size = (28,28)

    try:
        for file in files:
            f = await file.read()
            img = Image.open(io.BytesIO(f)).resize(new_size)
            pillow_images.append(img)
    except Exception:
        return html_open_tag + head_tag + body_center_open_tag + heading_exception_tag + form_prediction_tag + body_center_close_tag + html_close_tag

    names = [file.filename for file in files]

    for image, name in zip(pillow_images, names):
        path = 'static/' + name
        image.save(path)
        image_paths.append(path)

        predicted_class = predict(image)
        predictions.append(predicted_class)

    column_labels = ["Image", "Prediction"]

    return html_open_tag + head_tag + body_center_open_tag + marquee_prediction_tag + \
           table_tag(image_paths, predictions,
                     column_labels) + form_prediction_tag + body_center_close_tag + html_close_tag


# Prediction with TensorFlow Lite

def predict(img):

	img_converted = img.convert('L')
	img_array = np.array(img_converted)
	img_array = img_array.reshape(1, img_array.shape[0] * img_array.shape[1])
	img_array = img_array / 255 

	predicted_class = model.predict(img_array)
	predicted_class_integer = predicted_class.astype(np.int8)

	return classes[predicted_class_integer[0]]


# Static tags

html_open_tag = """<html>"""

head_tag = """<head><meta name="viewport" content="width=device-width, initial-scale=1"/></head>"""

heading = """<h3 style="font-family:Arial">The application will try to predict which of these categories they 
are:</h3> """

body_center_open_tag = """<body><center>"""

marquee_home_tag = """<marquee width="680" behavior="scroll"><h1 style="color:blue;font-family:Arial">Please upload 
your scenes</h1></marquee> """

form_tag = """
            <form action="/predictions/" enctype="multipart/form-data" method="post">
                <input name="files" type="file" multiple>
                <input type="submit">
            </form>
           """

body_center_close_tag = """</center></body>"""

html_close_tag = """</html>"""

marquee_prediction_tag = """<marquee width="680" behavior="scroll"><h1 style="color:blue;font-family:Arial">The 
predictions are as follows</h1></marquee> """

form_prediction_tag = """<br><form method="post" action="/"><button type="submit">Home</button></form>"""

heading_exception_tag = """<h3 style="font-family:Arial">Unable to process your request. Did you upload images?</h3>"""


# Dynamic tag

def table_tag(image_paths, classes_list, column_labels):
    s = '<table align="center">'

    if column_labels:
        s += '<tr><th><h4 style="font-family:Arial">' + column_labels[0] + '</h4></th> <th><h4 ' \
                                                                           'style="font-family:Arial">' + \
             column_labels[1] + '</h4></th></tr> '

    for name, image_path in zip(classes_list, image_paths):
        s += '<tr><td><img height="80" src="/' + image_path + '" ></td> <td style="text-align:center">' + name + '</td></tr>'

    s += '</table><br>'

    return s