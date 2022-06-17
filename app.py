# IMPORTS
from flask import Flask, request, render_template
import os
import shutil
import predictor

# FLASK
app = Flask(__name__)

# Set paths to upload folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
video_path=  os.path.join(os.path.join(APP_ROOT, 'static'),'temp')
wpath =  os.path.join(os.path.join(APP_ROOT, 'static'),'weight.hdf5')

# Main route
@app.route("/",methods=["GET","POST"])
def predict():

    # removing all other files from the temp folder
    shutil.rmtree(os.path.join(os.path.join(APP_ROOT, 'static'),'temp'))
    os.mkdir(os.path.join(os.path.join(APP_ROOT, 'static'),'temp'))
    video_disp=0
    ans=None
    filename=None
    if request.method == "POST":
        video_disp=1
        video = request.files['input_file']
        filename,ans = predictor.predict(video,video_path,wpath=wpath)
        filename='temp/'+filename
    return render_template("index.html", video_path = filename, video_disp=+video_disp, prediction = 'Prediction: '+str(ans))


if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)