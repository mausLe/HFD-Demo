from flask import Flask, render_template, request, session
from flask import flash, redirect, url_for

from werkzeug.utils import secure_filename
from datetime import timedelta
import os
from werkzeug.utils import secure_filename
import sys

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'mp4','png', 'jpg', 'jpeg', 'gif'}    

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def processing():
    if request.method == 'GET':
        return render_template(r'home.html')
    elif request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        # # if user does not select file, browser also
        # # submit an empty part without filename
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            # save file

            file.save(r'HFD-Demo\static\videos\input.mp4')
            # xử lí xong xuất file qua thư mục static và thay đổi giá trị img_test.jpg
            # khai báo static cho file
            # run_demo(r'path\uploads\input.mp4',r'static\output.webm',r'static\score.webm')
            # url_for(r'static', filename=r'output.webm',filename1=r'score.webm')
            return render_template(r'upload.html', file= r'output.webm',file1=r'score.webm' )
        return 'home'

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)

