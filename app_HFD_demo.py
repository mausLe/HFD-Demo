from flask import Flask, render_template, request, session
from flask import flash, redirect, url_for

from werkzeug.utils import secure_filename
import os, sys, time
import youtube_dl, shutil

import fall_detector_HFD_demo

f = fall_detector_HFD_demo.FallDetector()


app = Flask(__name__)
UPLOAD_FOLDER = r'\path\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

url = None
ALLOWED_EXTENSIONS = {'mp4','png', 'jpg', 'jpeg', 'gif'}    

myDict = {}
ydl = youtube_dl.YoutubeDL({'outtmpl': '%(id)s.%(ext)s'})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['GET', 'POST'])
def upload():

    global myDict
    if request.method == 'POST':
        print("REQUEST FORM: ", request.form)
        if request.form['submit_button'] == "BACK":
            return redirect(url_for('demo'))
    elif request.method == 'GET':
        return render_template('upload.html', content=myDict)
    
    return render_template('upload.html', content=myDict)


@app.route('/demo', methods=['GET', 'POST'])
def demo():

    global myDict
    myDict = {"input":r"static\videos\b2dl_fallingdown.mp4", "output":r"static\videos\b2dl_fallingdown.webm", "yt_link":None}

    # myDict = {"input":r"static\videos\fallingdown.mp4", "output":r"static\videos\out_fallingdown.webm", "yt_link":None}
    
    
    if request.method == 'GET':
        return render_template('demo.html', content=myDict)
    elif request.method == 'POST':
        # return request.form['fileSubmission']
        # check if the post request has the file part


        file = request.files['fileSubmission']

        yt_link = request.form["ytlink"]

        if yt_link:

            with ydl:
                result = ydl.extract_info(yt_link, download=True)

                #download = False - extract video info


            if "entries" in result:
                # resutl could be a playlist or a list of videos
                video = result["entries"][0]
            else:
                video = result
            
            # print("YT video: \n\n", video)
            video_url = video["webpage_url"]
            vid_title = video["title"]

            # vid_title = vid_title.replace("\\", "-")
            # vid_title = vid_title.replace(".", "_")
            # vid_title = vid_title.replace("/", "-")
            
            os.rename(video["id"]+".mp4", "input.mp4")
            dl_vid = r'input.mp4'

            dst_vid = r'static/videos/input.mp4'

            print("Destination video: ", dst_vid)
            print("video title", video_url)
            
            shutil.move(dl_vid, dst_vid)
            ydl.cache.remove()

            # myDict["input"] = dst_vid
            # myDict["output"] = dst_vid
        
        elif file:
            print("WORKING DIR: ", os.getcwd ())
            file.save(r'static/videos/input.mp4')
        # # if user does not select file, browser also
        # # submit an empty part without filename



        if file or yt_link:
            # myDict["output"] = r'static\videos\output.webm'
            # filename = secure_filename(file.filename)
            # save file
            
            # xử lí xong xuất file qua thư mục static và thay đổi giá trị img_test.jpg
            # khai báo static cho file
            # run_demo(r'path\uploads\input.mp4',r'static\output.webm',r'static\score.webm')
            # url_for(r'static', filename=r'output.webm',filename1=r'score.webm')
            
            # Run proccess to execute algorithms.py
            time.sleep(5)
            myDict["input"] = r'/static/videos/input.mp4'
            myDict["output"] = r'/static/videos/out.webm'

            f.change_parser_args(myDict)
            f.begin()

            time.sleep(5)

            return redirect(url_for('upload'))
    return render_template('demo.html', content=myDict)


@app.route('/return_url')
def return_url():

    return url

if __name__ == '__main__':
    app.run(debug=True)

