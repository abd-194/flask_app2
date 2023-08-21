from flask import Flask, render_template, Response, request, jsonify
import cv2
import pickle
import numpy as np
import pandas as pd
import threading
import io

from mediapipe.python.solutions import holistic as mp_holistic, drawing_utils as mp_drawing

app = Flask(__name__)
streaming_active = False
streaming_lock = threading.Lock()
camera = None

with open('models/body_language.pkl', 'rb') as f:
    model = pickle.load(f)

drawing_specs = [
    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
]

def generate_frames(camera):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while streaming_active:
            success, frame = camera.read()
            if not success:
                break
            else:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                results = holistic.process(image)
                image.flags.writeable = True

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, *drawing_specs)
                
                try:
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    X = pd.DataFrame([pose_row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]

                    if body_language_prob[np.argmax(body_language_prob)] > 0.98:
                        cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
                        cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                except:
                    pass

                ret, buffer = cv2.imencode('.jpg', image)
                frame_bytes = io.BytesIO(buffer.tobytes())
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes.read() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global streaming_active, camera
    streaming_active = True
    camera = cv2.VideoCapture(0)
    return Response(generate_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_stream')
def stop_stream():
    global streaming_active, camera
    with streaming_lock:
        streaming_active = False
        if camera is not None:
            camera.release()
            camera = None
    return jsonify({"message": "Stream stopped"})

if __name__ == '__main__':
    app.run(debug=True)
