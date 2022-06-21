import preprocess
import modelcode
from  werkzeug.utils import secure_filename
import os
def predict(video,video_upload_path):

        filenamev = secure_filename(video.filename)
        file_path = os.path.join(video_upload_path, filenamev)
        video.save(file_path)

        prediction_images = preprocess.preprocess_vdo(file_path,video_upload_path)
        base_model,model = modelcode.nn_model()
        prediction_images = base_model.predict(prediction_images)
        prediction_images = prediction_images.reshape(prediction_images.shape[0], 7*7*512)
        prediction = model.predict(prediction_images)
        pred  = prediction.mean()
        if pred>0.5:
            ans = 1
        else:
            ans = 0
        return filenamev,ans