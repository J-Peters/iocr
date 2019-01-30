from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
import json
import os
import re
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path

#import functions from tax form reader files
import base_functions as bf
import w8_processing as w8
import cv2

import PIL
from PIL import Image
import PIL.Image
from pytesseract import image_to_string
import pytesseract

# Set up Flask
app = Flask(__name__)
#app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

port = int(os.getenv('PORT', 8080))
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'pdf'])
def allowed_file(filename):
  return '.' in filename and \
                filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template('demo.html')

@app.route("/metadata/uploader", methods=['POST', 'GET'])
def upload_file():
   if request.method == 'POST':
      file = request.files['file']
      filename = file.filename
      # if user does not select file, browser also
      # submit an empty part without filename
      if file.filename == '':
         return 'No selected file'
      if file and allowed_file(file.filename):
         filename = secure_filename(file.filename)
         file.save(os.path.join(app.root_path, 'static', 'files', filename))
         stat_path = app.static_url_path.replace('/', '')
         return json.dumps({'file': os.path.join(stat_path, 'files', filename), 'size' : os.path.getsize(os.path.join(app.root_path, 'static', 'files', filename))})
   return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route("/execute", methods=['POST', 'GET'])
def process_file():
  if request.method == 'POST':
    content = request.json
    file = content['file']
    weightsPath=os.path.join('output', 'weights.hdf5')
    if content['w9f']:
       if content['hwd']:
          os.system('python base_functions.py --pdf-file ' + file + ' --load-model 1 --weights ' + weightsPath + ' > logfile.txt')
       else:
          os.system('python base_functions.py --pdf-file ' + file + ' > logfile.txt')
    elif content['w8f']:
       if content['hwd']:
          os.system('python w8_processing.py --pdf-file ' + file + ' --load-model 1 --weights ' + weightsPath + ' > logfile.txt')
       else:
          os.system('python w8_processing.py --pdf-file ' + file + ' > logfile.txt')
    f = open("logfile.txt", "r") #; print(f.read())
  return jsonify(f.read())

@app.route("/denoise", methods=['POST', 'GET'])
def remove_noise():
  if request.method == 'POST':
    file = request.files['file']
    filename = file.filename
    target = os.path.join(app.root_path, 'static', 'tmp', filename)
    file.save(target)
    bf.remove_noise(target, 'denoised-'+filename)
  return jsonify([os.path.join('static', 'tmp', filename), os.path.join('static', 'tmp', 'denoised-'+filename)])

@app.route("/deskew", methods=['POST', 'GET'])
def deskew():
  if request.method == 'POST':
    file = request.files['file']
    filename = file.filename
    target = os.path.join(app.root_path, 'static', 'tmp', filename)
    file.save(target)
    bf.rotate_doc(target, filename='deskewed-'+filename)
  return jsonify([os.path.join('static', 'tmp', filename), os.path.join('static', 'tmp', 'deskewed-'+filename)])

@app.route("/crop", methods=['POST', 'GET'])
def crop():
  if request.method == 'POST':
    file = request.files['file']
    filename = file.filename
    target = os.path.join(app.root_path, 'static', 'tmp', filename)
    file.save(target)
    bf.crop_form(target, filename='cropped-'+filename)
  return jsonify([os.path.join('static', 'tmp', filename), os.path.join('static', 'tmp', 'cropped-'+filename)])

@app.route("/markcheckboxes", methods=['POST', 'GET'])
def mark_checkboxes():
  if request.method == 'POST':
    file = request.files['file']
    filename = file.filename
    target = os.path.join(app.root_path, 'static', 'tmp', filename)
    file.save(target)
    form_c = cv2.resize(bf.crop_form(target), (2500, 3000))
    w8.checkbox(form_c, filename='checkbox-'+filename)
  return jsonify([os.path.join('static', 'tmp', filename), os.path.join('static', 'tmp', 'checkbox-'+filename)])

@app.route("/readdate", methods=['POST', 'GET'])
def read_date():
  if request.method == 'POST':
    file = request.files['file']
    filename = file.filename
    target = os.path.join(app.root_path, 'static', 'tmp', filename)
    file.save(target)
    weights = os.path.join('output', 'weights.hdf5')
    date_value = bf.read_date(target, weights, filename='hwdate-'+filename); #print(date_value)
  return jsonify([os.path.join('static', 'tmp', filename), os.path.join('static', 'tmp', 'hwdate-'+filename), date_value])

@app.route("/readtextboxes", methods=['POST', 'GET'])
def red_textboxes():
  if request.method == 'POST':
    file = request.files['file']
    filename = file.filename
    target = os.path.join(app.root_path, 'static', 'tmp', filename)
    file.save(target)
    img = cv2.imread(target, 0)
    idnum_img = bf.get_tin(img, filename='textbox-'+filename)
    idnum = pytesseract.image_to_string((Image.fromarray(idnum_img)).convert("RGB"), lang='eng')
    if re.match('[0-9]*\s*-[0-9]*\s*-[0-9]*\s*', idnum):
       idNum = idnum.replace(' ', '')
    else:
       idNum = idnum.replace(' ', '')
  return jsonify([os.path.join('static', 'tmp', filename), os.path.join('static', 'tmp', 'textbox-'+filename), idNum])


if __name__ == "__main__": app.run(host='0.0.0.0', port=port, debug=True)

