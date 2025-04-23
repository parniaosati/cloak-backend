from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

background = None
cloak_color = None

HSV_RANGES = {
    'red':    [(np.array([0, 70, 50]), np.array([10, 255, 255])),
               (np.array([170, 70, 50]), np.array([180, 255, 255]))],
    'green':  [(np.array([35, 70, 50]), np.array([85, 255, 255]))],
    'blue':   [(np.array([90, 70, 50]), np.array([130, 255, 255]))],
    'yellow': [(np.array([20, 100, 100]), np.array([35, 255, 255]))]
}

@app.route('/ready', methods=['POST'])
def set_background():
    global background
    data = request.json
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    background = cv2.flip(frame, 1)
    print("✅ Background captured!")
    return jsonify({"status": "background captured"})

@app.route('/set-color', methods=['POST'])
def set_color():
    global cloak_color
    data = request.json
    cloak_color = data.get("color")
    print(f"🎨 Color selected: {cloak_color}")
    return jsonify({"status": "color set"})

@app.route('/process', methods=['POST'])
def process_frame():
    global background, cloak_color

    if background is None:
        print("🚫 Background not set yet.")
    if cloak_color is None:
        print("🚫 Cloak color not chosen yet.")
    if background is None or cloak_color is None:
        return jsonify({'image': None})

    data = request.json
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros_like(hsv[:, :, 0])
    for lower, upper in HSV_RANGES.get(cloak_color, []):
        mask |= cv2.inRange(hsv, lower, upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    inv_mask = cv2.bitwise_not(mask)

    res1 = cv2.bitwise_and(background, background, mask=mask)
    res2 = cv2.bitwise_and(frame, frame, mask=inv_mask)
    final = cv2.add(res1, res2)

    _, buffer = cv2.imencode('.jpg', final)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': f'data:image/jpeg;base64,{encoded_image}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
