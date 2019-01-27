from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
import json
import os
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
#import functions from tax form reader files
import base_functions as bf
import w8_processing as w8
import cv2

# Set up Flask
app = Flask(__name__)
#app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

port = int(os.getenv('PORT', 8000))
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
         #flash('No selected file')
         #return redirect(request.url)
         return 'No selected file'
      if file and allowed_file(file.filename):
         filename = secure_filename(file.filename)
         file.save(os.path.join(app.root_path, 'static', 'files', filename))
         stat_path = app.static_url_path.replace('/', '')
         #os.system('python base_functions.py --pdf-file D:\\DB_OCR\\deutschBank\\USB-0134388.pdf')
         #return redirect(url_for('home'))
         #print(os.path.getsize(os.path.join(app.root_path, 'static/files', filename)))
         #return os.path.join(app.static_url_path, 'files', filename)
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
    content = request.json; print(content)
    file = content['file']; print(file)
    if content['w9f']:
       if content['hwd']:
          os.system('python base_functions.py --pdf-file ' + file + ' --load-model 1 --weights output\weights.hdf5 > logfile.txt')
       else:
          os.system('python base_functions.py --pdf-file ' + file + ' > logfile.txt')
    elif content['w8f']:
       if content['hwd']:
          os.system('python w8_processing.py --pdf-file ' + file + ' --load-model 1 --weights output\weights.hdf5 > logfile.txt')
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
    bf.remove_noise(target, 'denoised-'+filename)#; print(os.path.join(app.root_path, 'static', 'tmp', 'denoised-'+filename))
  return jsonify([os.path.join('static', 'tmp', filename), os.path.join('static', 'tmp', 'denoised-'+filename)])

@app.route("/deskew", methods=['POST', 'GET'])
def deskew():
  if request.method == 'POST':
    file = request.files['file']
    filename = file.filename
    target = os.path.join(app.root_path, 'static', 'tmp', filename)
    file.save(target)
    bf.rotate_doc(target, filename='deskewed-'+filename)#; print(os.path.join(app.root_path, 'static', 'tmp', 'deskewed-'+filename))
  return jsonify([os.path.join('static', 'tmp', filename), os.path.join('static', 'tmp', 'deskewed-'+filename)])

@app.route("/crop", methods=['POST', 'GET'])
def crop():
  if request.method == 'POST':
    file = request.files['file']
    filename = file.filename
    target = os.path.join(app.root_path, 'static', 'tmp', filename)
    file.save(target)
    bf.crop_form(target, filename='cropped-'+filename)#; print(os.path.join(app.root_path, 'static', 'tmp', 'cropped-'+filename))
  return jsonify([os.path.join('static', 'tmp', filename), os.path.join('static', 'tmp', 'cropped-'+filename)])

@app.route("/markcheckboxes", methods=['POST', 'GET'])
def mark_checkboxes():
  if request.method == 'POST':
    file = request.files['file']
    filename = file.filename
    target = os.path.join(app.root_path, 'static', 'tmp', filename)
    file.save(target)
    form_c = cv2.resize(bf.crop_form(target), (2500, 3000))
    w8.checkbox(form_c, filename='checkbox-'+filename)#; print(os.path.join(app.root_path, 'static', 'tmp', 'checkbox-'+filename))
  return jsonify([os.path.join('static', 'tmp', filename), os.path.join('static', 'tmp', 'checkbox-'+filename)])

if __name__ == "__main__": app.run(host='0.0.0.0', port=port, debug=True)

