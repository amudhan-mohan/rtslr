from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque, Counter
import threading
import time
import json

app = Flask(__name__)

# ==========================================
# Model Architecture
# ==========================================
class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=258, hidden_size=128, num_layers=2, num_classes=5, dropout=0.5):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        final_out = self.dropout(out[:, -1, :])
        return self.fc(final_out)

# ==========================================
# Configuration
# ==========================================
CLASSES = [
    "hello", "bye", "again", "you", "deaf", "hearing", "teacher",
    "thank_you", "welcome", "sorry", "correct", "wrong", "good",
    "morning", "understand", "yes", "no", "exam", "home", "work",
    "can", "will", "do", "your", "write", "this", "it", "difficult",
    "read", "yesterday", "my", "name", "who", "what", "class",
    "leader", "i_or_me", "i_dont_understand", "tomorrow", "how",
    "sit_down", "stand_up", "come", "study", "look", "here",
    "now", "help", "quiet"
]

SEQUENCE_LENGTH = 30
KEYPOINT_DIM = 258
MODEL_PATH = 'sign_lstm_best.pt'
EMA_ALPHA = 0.5

# ==========================================
# Global State
# ==========================================
state = {
    "current_prediction": "Waiting...",
    "confidence": 0.0,
    "all_probs": {},
    "history": [],
    "session_counts": {},
    "total_predictions": 0,
    "running": True,
    "camera_active": False,  # Camera starts OFF
}
state_lock = threading.Lock()

# Global camera object
cap = None
cap_lock = threading.Lock()

# ==========================================
# Load Model
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model = SignLanguageLSTM(input_size=KEYPOINT_DIM, num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    MODEL_LOADED = True
    print(f"✅ Model loaded on {device}")
except Exception as e:
    MODEL_LOADED = False
    print(f"⚠️ Model not loaded: {e}. Running in demo mode.")

# MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# Helper Functions
# ==========================================
def extract_keypoints(pose_results, hand_results):
    if pose_results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in pose_results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)

    lh, rh = np.zeros(21 * 3), np.zeros(21 * 3)
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_lms, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            label = handedness.classification[0].label
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]).flatten()
            if label == 'Left':
                lh = coords
            else:
                rh = coords

    return np.concatenate([pose, lh, rh])

def normalize_frame(frame):
    frame = frame.copy()
    pose = frame[0:132].reshape(33, 4)
    left_hip, right_hip = pose[23, :3], pose[24, :3]
    left_shoulder, right_shoulder = pose[11, :3], pose[12, :3]

    if np.any(left_hip) and np.any(right_hip):
        center = (left_hip + right_hip) / 2.0
        shoulder_dist = max(np.linalg.norm(left_shoulder - right_shoulder), 0.01)
        for j in range(33):
            pose[j, :3] = (pose[j, :3] - center) / shoulder_dist
        frame[0:132] = pose.flatten()

    lh = frame[132:195].reshape(21, 3)
    if np.any(lh):
        lh_center = lh[0].copy()
        lh_scale = max(np.linalg.norm(lh[0] - lh[9]), 0.01)
        for j in range(21):
            lh[j] = (lh[j] - lh_center) / lh_scale
        frame[132:195] = lh.flatten()

    rh = frame[195:258].reshape(21, 3)
    if np.any(rh):
        rh_center = rh[0].copy()
        rh_scale = max(np.linalg.norm(rh[0] - rh[9]), 0.01)
        for j in range(21):
            rh[j] = (rh[j] - rh_center) / rh_scale
        frame[195:258] = rh.flatten()

    return frame

def apply_ema(current_lms, prev_lms, alpha):
    if prev_lms is None:
        return current_lms
    return current_lms * alpha + prev_lms * (1 - alpha)

# ==========================================
# Video Generator
# ==========================================
def generate_frames():
    global cap
    
    sequence = deque(maxlen=SEQUENCE_LENGTH)
    prediction_buffer = deque(maxlen=20)
    previous_landmarks = None
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        
        while state["running"]:
            # Check camera state
            with state_lock:
                camera_active = state["camera_active"]
            
            # Handle camera initialization when turned ON
            if camera_active and cap is None:
                with cap_lock:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        print("❌ Failed to open camera")
                        with state_lock:
                            state["camera_active"] = False
                            camera_active = False
                    else:
                        print("✅ Camera started successfully")
            
            # Handle camera release when turned OFF
            if not camera_active and cap is not None:
                with cap_lock:
                    cap.release()
                    cap = None
                    print("📷 Camera stopped")
            
            # If camera is OFF or not available, show placeholder
            if not camera_active or cap is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "CAMERA OFF", (220, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 106, 130), 2)
                cv2.putText(frame, "Click 'Turn On' to start", (200, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 106, 130), 1)
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.05)
                continue
            
            # Camera is ON - capture frame
            with cap_lock:
                if cap is None:
                    continue
                ret, frame = cap.read()
            
            if not ret:
                time.sleep(0.01)
                continue
            
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            
            pose_res = pose.process(img_rgb)
            hand_res = hands.process(img_rgb)
            img_rgb.flags.writeable = True
            
            # Draw landmarks
            if pose_res.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 150), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2)
                )
            if hand_res.multi_hand_landmarks:
                for hand_landmarks in hand_res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2)
                    )
            
            # Keypoint pipeline
            keypoints = extract_keypoints(pose_res, hand_res)
            normalized_keypoints = normalize_frame(keypoints)
            smoothed_keypoints = apply_ema(normalized_keypoints, previous_landmarks, EMA_ALPHA)
            previous_landmarks = smoothed_keypoints
            sequence.append(smoothed_keypoints)
            
            local_prediction = "Waiting..."
            local_confidence = 0.0
            all_probs = {}
            
            if len(sequence) == SEQUENCE_LENGTH and MODEL_LOADED:
                seq_array = np.array(sequence)
                motion_variance = np.var(seq_array, axis=0).mean()
                
                if motion_variance < 0.001:
                    local_prediction = "Idle"
                    local_confidence = 0.0
                else:
                    input_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        res = model(input_tensor)
                        probs = torch.softmax(res, dim=1).cpu().numpy()[0]
                    
                    max_idx = np.argmax(probs)
                    local_confidence = float(probs[max_idx])
                    all_probs = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
                    
                    if local_confidence > 0.85:
                        local_prediction = CLASSES[max_idx]
                    
                    prediction_buffer.append(local_prediction)
                    
                    if len(prediction_buffer) > 0:
                        vote_tally = Counter(prediction_buffer).most_common(1)[0]
                        winning_gesture, winning_votes = vote_tally
                        if winning_votes >= 12:
                            with state_lock:
                                prev = state["current_prediction"]
                                state["current_prediction"] = winning_gesture
                                state["confidence"] = local_confidence
                                state["all_probs"] = all_probs
                                
                                if winning_gesture not in ["Idle", "Waiting...", "Thinking..."] and winning_gesture != prev:
                                    entry = {
                                        "gesture": winning_gesture,
                                        "confidence": round(local_confidence * 100, 1),
                                        "timestamp": time.strftime("%H:%M:%S")
                                    }
                                    state["history"].insert(0, entry)
                                    state["history"] = state["history"][:50]
                                    state["session_counts"][winning_gesture] = state["session_counts"].get(winning_gesture, 0) + 1
                                    state["total_predictions"] += 1
            
            # Encode and yield frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Cleanup
    if cap is not None:
        cap.release()

# ==========================================
# Routes
# ==========================================
@app.route('/')
def index():
    return render_template('index.html', classes=CLASSES, model_loaded=MODEL_LOADED)

@app.route('/inference')
def inference():
    return render_template('inference.html', classes=CLASSES, model_loaded=MODEL_LOADED)

@app.route('/session-analytics')
def sessionanalytics():
    return render_template('analytics.html', classes=CLASSES)

@app.route('/dataset-analytics')
def datasetanalytics():
    return render_template('data_analytics.html', classes=CLASSES)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def get_state():
    with state_lock:
        return jsonify({
            "prediction": state["current_prediction"],
            "confidence": round(state["confidence"] * 100, 1),
            "all_probs": {k: round(v * 100, 1) for k, v in state["all_probs"].items()},
            "history": state["history"][:10],
            "total_predictions": state["total_predictions"],
            "session_counts": state["session_counts"],
            "running": state["running"],
        })

@app.route('/api/analytics')
def get_analytics():
    with state_lock:
        counts = state["session_counts"]
        total = state["total_predictions"]
        history = state["history"]

        if history:
            avg_conf = sum(h["confidence"] for h in history) / len(history)
        else:
            avg_conf = 0.0

        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return jsonify({
            "total_predictions": total,
            "unique_gestures": len(counts),
            "avg_confidence": round(avg_conf, 1),
            "top_gestures": [{"gesture": g, "count": c, "pct": round(c/max(total,1)*100,1)} for g, c in top],
            "history": history[:20],
            "all_counts": counts,
        })

@app.route('/api/camera/control', methods=['POST'])
def camera_control():
    """Control camera state (on/off)"""
    try:
        data = request.get_json()
        camera_on = data.get('camera_on', False)
        
        with state_lock:
            state["camera_active"] = camera_on
            print(f"📷 Camera state set to: {'ON' if camera_on else 'OFF'}")
        
        return jsonify({
            "success": True,
            "camera_on": camera_on,
            "message": f"Camera turned {'ON' if camera_on else 'OFF'} successfully"
        })
        
    except Exception as e:
        print(f"Error in camera_control: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    """Clear recognition history"""
    try:
        with state_lock:
            state["history"] = []
            state["session_counts"] = {}
            state["total_predictions"] = 0
            state["current_prediction"] = "Waiting..."
            state["confidence"] = 0.0
            state["all_probs"] = {}
        
        return jsonify({
            "success": True,
            "message": "History cleared successfully"
        })
        
    except Exception as e:
        print(f"Error in clear_history: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)