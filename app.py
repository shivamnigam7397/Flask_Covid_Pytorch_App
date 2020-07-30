import os

from flask import Flask, render_template, request, redirect,url_for
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename
from inference import prediction_image,loader,model_new


model_new.eval()
app = Flask(__name__)

# UPLOAD_FOLDER = 'C:/Users/lenovo/Desktop/flask_covid_ct/static/images/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        loaded_image_file = request.files.get('file')
        if not loaded_image_file:
            return
        loaded_image = loaded_image_file.read()
        category = prediction_image(model = model_new, name_image = loaded_image)
        return render_template('result.html', Disease=category)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
