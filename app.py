from flask import Flask, render_template, request, session
from flask import flash, redirect, url_for

from werkzeug.utils import secure_filename
import os, sys

app = Flask(__name__)
UPLOAD_FOLDER = r'\path\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'mp4','png', 'jpg', 'jpeg', 'gif'}    

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    myDict = {"input":None, "output":None, "yt_link":None}
    if request.method == 'GET':
        return render_template('demo.html')
    elif request.method == 'POST':
        # return request.form['fileSubmission']
        # check if the post request has the file part
        file = request.files['fileSubmission']

        myDict = file


        # # if user does not select file, browser also
        # # submit an empty part without filename

        if file:
            # filename = secure_filename(file.filename)
            # save file

            file.save(r'HFD-Demo\static\videos\input.mp4')
            # xử lí xong xuất file qua thư mục static và thay đổi giá trị img_test.jpg
            # khai báo static cho file
            # run_demo(r'path\uploads\input.mp4',r'static\output.webm',r'static\score.webm')
            # url_for(r'static', filename=r'output.webm',filename1=r'score.webm')
            return render_template('upload.html', content=myDict)
    return render_template('demo.html')

if __name__ == '__main__':
    app.run(debug=True)

