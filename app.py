from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import torch
import torch.nn as nn
import threading
import time
import os
import sys
from collections import deque, Counter

# ── JAX compatibility patch ──────────────────────────────────────────
try:
    import ml_dtypes
    class MLDtypesMock:
        def __init__(self, original_module):
            self.original_module = original_module
            self.fallback = np.int32
        def __getattr__(self, name):
            if hasattr(self.original_module, name):
                return getattr(self.original_module, name)
            return self.fallback
    sys.modules['ml_dtypes'] = MLDtypesMock(ml_dtypes)
except ImportError:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Safe imports ───────────────────────────────────────────────────────
JOBLIB_AVAILABLE = False
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    print("⚠️ joblib not installed. Install with: pip install joblib")

MEDIAPIPE_AVAILABLE = False
mp_pose = None
mp_hands = None
mp_holistic = None
mp_drawing = None

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    print("✅ MediaPipe imported successfully")
except ImportError as e:
    print(f"⚠️ MediaPipe not available: {e}")

app = Flask(__name__)

# ==========================================
# DYNAMIC MODE CONFIG (ISL Gestures - LSTM)
# ==========================================
class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=258, hidden_size=128, num_layers=2, num_classes=49, dropout=0.5):
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        final_out = self.dropout(out[:, -1, :])
        return self.fc(final_out)

DYNAMIC_CLASSES = [
    "hello", "bye", "again", "you", "deaf", "hearing", "teacher",
    "thank_you", "welcome", "sorry", "correct", "wrong", "good",
    "morning", "understand", "yes", "no", "exam", "home", "work",
    "can", "will", "do", "your", "write", "this", "it", "difficult",
    "read", "yesterday", "my", "name", "who", "what", "class",
    "leader", "i_or_me", "i_dont_understand", "tomorrow", "how",
    "sit_down", "stand_up", "come", "study", "look", "here",
    "now", "help", "quiet"
]

DYNAMIC_SEQUENCE_LENGTH = 30
DYNAMIC_KEYPOINT_DIM = 258
DYNAMIC_MODEL_PATH = 'sign_lstm_best.pt'
DYNAMIC_EMA_ALPHA = 0.5
DYNAMIC_CONFIDENCE_THRESHOLD = 0.85
DYNAMIC_VOTING_THRESHOLD = 12
DYNAMIC_BUFFER_SIZE = 20

# ==========================================
# STATIC MODE CONFIG (Letters - sklearn)
# ==========================================
STATIC_MODEL_PATH = 'isl_alphabet_model.pkl'
STATIC_VOTING_BUFFER_SIZE = 8
STATIC_MIN_VOTES = 4
STATIC_CONFIDENCE_THRESHOLD = 65
STATIC_CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + [str(i) for i in range(10)]

# ==========================================
# GLOBAL STATE
# ==========================================
# Dynamic mode state
dynamic_state = {
    "current_prediction": "Waiting...",
    "confidence": 0.0,
    "all_probs": {},
    "history": [],
    "session_counts": {},
    "total_predictions": 0,
    "running": True,
    "camera_active": False,
}

# Static mode state
static_state = {
    "letter": "—",
    "confidence": 0.0,
    "all_probs": {},
    "hand_present": False,
    "history": [],
    "counts": {},
    "total": 0,
    "word": [],
    "sentence": [],
    "camera_active": False,
}

state_lock = threading.Lock()
dynamic_lock = threading.Lock()
static_lock = threading.Lock()

# Global camera objects
dynamic_cap = None
static_cap = None
cap_lock = threading.Lock()

# ==========================================
# LOAD MODELS
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DYNAMIC_MODEL_LOADED = False
dynamic_model = None

if os.path.exists(DYNAMIC_MODEL_PATH):
    try:
        dynamic_model = SignLanguageLSTM(input_size=DYNAMIC_KEYPOINT_DIM, num_classes=len(DYNAMIC_CLASSES)).to(device)
        dynamic_model.load_state_dict(torch.load(DYNAMIC_MODEL_PATH, map_location=device, weights_only=True))
        dynamic_model.eval()
        DYNAMIC_MODEL_LOADED = True
        print(f"✅ Dynamic model loaded successfully on {device}")
    except Exception as e:
        print(f"⚠️ Dynamic model not loaded: {e}. Running in demo mode.")
else:
    print(f"⚠️ Dynamic model file not found at {DYNAMIC_MODEL_PATH}. Running in demo mode.")

# Load static model
STATIC_MODEL_LOADED = False
clf = None

if JOBLIB_AVAILABLE and os.path.exists(STATIC_MODEL_PATH):
    try:
        clf = joblib.load(STATIC_MODEL_PATH)
        STATIC_MODEL_LOADED = True
        print(f"✅ Static model loaded — {len(STATIC_CLASSES)} classes")
    except Exception as e:
        print(f"⚠️ Static model load failed ({e}). Running in demo mode.")
else:
    if not os.path.exists(STATIC_MODEL_PATH):
        print(f"⚠️ Static model file not found: {STATIC_MODEL_PATH}. Running in demo mode.")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def extract_keypoints(pose_results, hand_results):
    """Extract pose and hand keypoints for dynamic mode"""
    if not MEDIAPIPE_AVAILABLE:
        return np.zeros(DYNAMIC_KEYPOINT_DIM)
    
    try:
        if pose_results and hasattr(pose_results, 'pose_landmarks') and pose_results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                             for lm in pose_results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33 * 4)

        lh, rh = np.zeros(21 * 3), np.zeros(21 * 3)
        if hand_results and hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_lms, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                label = handedness.classification[0].label
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]).flatten()
                if label == 'Left':
                    lh = coords
                else:
                    rh = coords

        return np.concatenate([pose, lh, rh])
    except Exception as e:
        print(f"Keypoint extraction error: {e}")
        return np.zeros(DYNAMIC_KEYPOINT_DIM)

def normalize_frame(frame):
    """Normalize keypoints for dynamic mode"""
    try:
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
    except Exception as e:
        return frame

def apply_ema(current_lms, prev_lms, alpha):
    if prev_lms is None:
        return current_lms
    return current_lms * alpha + prev_lms * (1 - alpha)

def extract_features(results):
    """Extract features for static mode"""
    pose = (np.array([[r.x, r.y, r.z, r.visibility]
                       for r in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 4))

    def get_hand(hand_lms):
        if not hand_lms:
            return np.zeros(21 * 3 + 5)
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
        relative = (coords - coords[0]).flatten()
        tips = [4, 8, 12, 16, 20]
        distances = [np.linalg.norm(coords[t] - coords[0]) for t in tips]
        return np.concatenate([relative, distances])

    lh = get_hand(results.left_hand_landmarks)
    rh = get_hand(results.right_hand_landmarks)
    return np.concatenate([pose, lh, rh])

# ==========================================
# VIDEO GENERATORS
# ==========================================
def generate_dynamic_frames():
    """Video generator for dynamic mode (ISL gestures)"""
    global dynamic_cap
    
    sequence = deque(maxlen=DYNAMIC_SEQUENCE_LENGTH)
    prediction_buffer = deque(maxlen=DYNAMIC_BUFFER_SIZE)
    previous_landmarks = None
    
    demo_mode = not DYNAMIC_MODEL_LOADED
    
    pose_ctx = None
    hands_ctx = None
    
    if MEDIAPIPE_AVAILABLE and mp_pose and mp_hands:
        pose_ctx = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        hands_ctx = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        pose = pose_ctx.__enter__()
        hands = hands_ctx.__enter__()
    else:
        pose = None
        hands = None
    
    try:
        while dynamic_state["running"]:
            try:
                with dynamic_lock:
                    camera_active = dynamic_state["camera_active"]
                
                if camera_active and dynamic_cap is None:
                    with cap_lock:
                        dynamic_cap = cv2.VideoCapture(0)
                        if not dynamic_cap.isOpened():
                            print("❌ Failed to open dynamic camera")
                            with dynamic_lock:
                                dynamic_state["camera_active"] = False
                                camera_active = False
                        else:
                            print("✅ Dynamic camera started")
                
                if not camera_active and dynamic_cap is not None:
                    with cap_lock:
                        dynamic_cap.release()
                        dynamic_cap = None
                        print("📷 Dynamic camera stopped")
                        sequence.clear()
                        prediction_buffer.clear()
                        previous_landmarks = None
                
                if not camera_active or dynamic_cap is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "CAMERA OFF", (220, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 106, 130), 2)
                    cv2.putText(frame, "Click 'Turn On' to start", (200, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 106, 130), 1)
                    
                    if demo_mode:
                        cv2.putText(frame, "DEMO MODE - Model not loaded", (180, 320), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1)
                    
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    time.sleep(0.05)
                    continue
                
                with cap_lock:
                    if dynamic_cap is None:
                        continue
                    ret, frame = dynamic_cap.read()
                
                if not ret:
                    time.sleep(0.01)
                    continue
                
                frame = cv2.flip(frame, 1)
                
                pose_res = None
                hand_res = None
                
                if pose and hands and MEDIAPIPE_AVAILABLE:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_rgb.flags.writeable = False
                    pose_res = pose.process(img_rgb)
                    hand_res = hands.process(img_rgb)
                    img_rgb.flags.writeable = True
                    
                    if pose_res and pose_res.pose_landmarks and mp_drawing:
                        mp_drawing.draw_landmarks(
                            frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 150), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2)
                        )
                    if hand_res and hand_res.multi_hand_landmarks and mp_drawing:
                        for hand_landmarks in hand_res.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=2)
                            )
                
                if not demo_mode and MEDIAPIPE_AVAILABLE:
                    try:
                        keypoints = extract_keypoints(pose_res, hand_res)
                        normalized_keypoints = normalize_frame(keypoints)
                        smoothed_keypoints = apply_ema(normalized_keypoints, previous_landmarks, DYNAMIC_EMA_ALPHA)
                        previous_landmarks = smoothed_keypoints
                        sequence.append(smoothed_keypoints)
                        
                        if len(sequence) == DYNAMIC_SEQUENCE_LENGTH:
                            seq_array = np.array(sequence)
                            motion_variance = np.var(seq_array, axis=0).mean()
                            
                            if motion_variance < 0.001:
                                local_prediction = "Idle"
                                local_confidence = 0.0
                            else:
                                input_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    res = dynamic_model(input_tensor)
                                    probs = torch.softmax(res, dim=1).cpu().numpy()[0]
                                
                                max_idx = np.argmax(probs)
                                local_confidence = float(probs[max_idx])
                                all_probs = {DYNAMIC_CLASSES[i]: float(probs[i]) for i in range(len(DYNAMIC_CLASSES))}
                                
                                if local_confidence > DYNAMIC_CONFIDENCE_THRESHOLD:
                                    local_prediction = DYNAMIC_CLASSES[max_idx]
                                
                                prediction_buffer.append(local_prediction)
                                
                                if len(prediction_buffer) > 0:
                                    vote_tally = Counter(prediction_buffer).most_common(1)[0]
                                    winning_gesture, winning_votes = vote_tally
                                    if winning_votes >= DYNAMIC_VOTING_THRESHOLD:
                                        with dynamic_lock:
                                            prev = dynamic_state["current_prediction"]
                                            dynamic_state["current_prediction"] = winning_gesture
                                            dynamic_state["confidence"] = local_confidence
                                            dynamic_state["all_probs"] = all_probs
                                            
                                            if winning_gesture not in ["Idle", "Waiting...", "Thinking..."] and winning_gesture != prev:
                                                entry = {
                                                    "gesture": winning_gesture,
                                                    "confidence": round(local_confidence * 100, 1),
                                                    "timestamp": time.strftime("%H:%M:%S")
                                                }
                                                dynamic_state["history"].insert(0, entry)
                                                dynamic_state["history"] = dynamic_state["history"][:50]
                                                dynamic_state["session_counts"][winning_gesture] = dynamic_state["session_counts"].get(winning_gesture, 0) + 1
                                                dynamic_state["total_predictions"] += 1
                    except Exception as e:
                        print(f"⚠️ Dynamic inference error: {e}")
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
            except Exception as e:
                print(f"❌ Dynamic frame generation error: {e}")
                continue
    
    finally:
        if pose_ctx:
            pose_ctx.__exit__(None, None, None)
        if hands_ctx:
            hands_ctx.__exit__(None, None, None)
        if dynamic_cap is not None:
            dynamic_cap.release()

def generate_static_frames():
    """Video generator for static mode (Letters)"""
    global static_cap
    
    prediction_history = deque(maxlen=STATIC_VOTING_BUFFER_SIZE)
    holistic = mp_holistic.Holistic(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) if MEDIAPIPE_AVAILABLE and mp_holistic else None
    
    last_letter = None
    last_log_time = 0
    last_stable_letter = None

    try:
        while True:
            with static_lock:
                camera_should_run = static_state["camera_active"]
            
            if camera_should_run:
                with cap_lock:
                    if static_cap is None or not static_cap.isOpened():
                        static_cap = cv2.VideoCapture(0)
                        if not static_cap.isOpened():
                            print("❌ Failed to open static camera")
                            with static_lock:
                                static_state["camera_active"] = False
                            err = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(err, "CAMERA NOT FOUND", (140,240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                            _, buf = cv2.imencode('.jpg', err)
                            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                            time.sleep(0.5)
                            continue
                        static_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        static_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        print("✅ Static camera opened")
                
                with cap_lock:
                    ret, frame = (static_cap.read() if static_cap and static_cap.isOpened() else (False, None))
                
                if ret:
                    frame = cv2.flip(frame, 1)
                    
                    if holistic and MEDIAPIPE_AVAILABLE:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_rgb.flags.writeable = False
                        results = holistic.process(img_rgb)
                        img_rgb.flags.writeable = True

                        if results.left_hand_landmarks and mp_drawing:
                            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255,200,0), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(255,120,0), thickness=2))
                        if results.right_hand_landmarks and mp_drawing:
                            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,220,255), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(0,160,255), thickness=2))
                    else:
                        results = None

                    letter, confidence, all_probs = "—", 0.0, {}
                    hand_present = bool(results and (results.left_hand_landmarks or results.right_hand_landmarks)) if results else False
                    
                    if hand_present and STATIC_MODEL_LOADED and results:
                        try:
                            feats = extract_features(results).reshape(1, -1)
                            raw = clf.predict(feats)[0]
                            probs = clf.predict_proba(feats)[0]
                            confidence = float(np.max(probs)) * 100
                            all_probs = {str(cls): round(float(p)*100, 1) for cls, p in zip(clf.classes_, probs)}
                            
                            if confidence >= STATIC_CONFIDENCE_THRESHOLD:
                                prediction_history.append(raw)
                                tally = Counter(prediction_history).most_common(1)[0]
                                if tally[1] >= STATIC_MIN_VOTES:
                                    letter = str(tally[0])
                                    last_stable_letter = letter
                                
                                now = time.time()
                                if letter != "—" and (letter != last_letter or now - last_log_time > 1.0):
                                    if letter != last_letter:
                                        last_letter, last_log_time = letter, now
                                        with static_lock:
                                            entry = {"letter": letter, "conf": round(confidence,1), "time": time.strftime("%H:%M:%S")}
                                            static_state["history"].insert(0, entry)
                                            static_state["history"] = static_state["history"][:60]
                                            static_state["counts"][letter] = static_state["counts"].get(letter, 0) + 1
                                            static_state["total"] += 1
                        except Exception as e:
                            print(f"⚠️ Static inference error: {e}")
                    else:
                        prediction_history.clear()
                        if not hand_present:
                            last_letter = None
                            last_stable_letter = None

                    with static_lock:
                        static_state["letter"] = last_stable_letter if last_stable_letter else "—"
                        static_state["confidence"] = confidence
                        static_state["all_probs"] = all_probs
                        static_state["hand_present"] = hand_present

                    if last_stable_letter and last_stable_letter != "—":
                        cv2.putText(frame, last_stable_letter, (20, 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3, cv2.LINE_AA)

                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                else:
                    err = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(err, "Capture error", (200,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    _, buf = cv2.imencode('.jpg', err)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                    time.sleep(0.05)
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "CAMERA OFF", (220,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (90,106,130), 2)
                cv2.putText(placeholder, "Click 'Turn On' to start", (180,280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90,106,130), 1)
                _, buf = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                time.sleep(0.05)
    finally:
        with cap_lock:
            if static_cap:
                static_cap.release()
                static_cap = None
        if holistic:
            holistic.close()

# ==========================================
# ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html', dynamic_loaded=DYNAMIC_MODEL_LOADED,
                          static_loaded=STATIC_MODEL_LOADED,
                          dynamic_classes=len(DYNAMIC_CLASSES),
                          static_classes=len(STATIC_CLASSES))

@app.route('/change-mode')
def change_mode():
    """Change Mode page - choose between Dynamic and Static"""
    return render_template('change-mode.html', 
                          dynamic_loaded=DYNAMIC_MODEL_LOADED,
                          static_loaded=STATIC_MODEL_LOADED,
                          dynamic_classes=len(DYNAMIC_CLASSES),
                          static_classes=len(STATIC_CLASSES))

@app.route('/dynamic/inference')
def dynamic_inference():
    return render_template('inference.html', classes=DYNAMIC_CLASSES, model_loaded=DYNAMIC_MODEL_LOADED)

@app.route('/dynamic/session-analytics')
def dynamic_session_analytics():
    return render_template('analytics.html', classes=DYNAMIC_CLASSES)

@app.route('/dynamic/dataset-analytics')
def dynamic_dataset_analytics():
    return render_template('data_analytics.html', classes=DYNAMIC_CLASSES)

@app.route('/static/inference')
def static_inference():
    return render_template('static_inference.html')

@app.route('/static/session-analytics')
def static_session_analytics():
    return render_template('static_analytics.html')

@app.route('/dynamic/video_feed')
def dynamic_video_feed():
    return Response(generate_dynamic_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/video_feed')
def static_video_feed():
    return Response(generate_static_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dynamic/api/state')
def dynamic_get_state():
    with dynamic_lock:
        return jsonify({
            "prediction": dynamic_state["current_prediction"],
            "confidence": round(dynamic_state["confidence"] * 100, 1),
            "all_probs": {k: round(v * 100, 1) for k, v in dynamic_state["all_probs"].items()},
            "history": dynamic_state["history"][:10],
            "total_predictions": dynamic_state["total_predictions"],
            "session_counts": dynamic_state["session_counts"],
            "running": dynamic_state["running"],
            "model_loaded": DYNAMIC_MODEL_LOADED
        })

@app.route('/static/api/state')
def static_get_state():
    with static_lock:
        return jsonify({
            "letter": static_state["letter"],
            "confidence": round(static_state["confidence"], 1),
            "all_probs": static_state["all_probs"],
            "hand_present": static_state["hand_present"],
            "history": static_state["history"][:12],
            "total": static_state["total"],
            "unique": len(static_state["counts"]),
            "session_counts": static_state["counts"],
        })

@app.route('/dynamic/api/analytics')
def dynamic_get_analytics():
    with dynamic_lock:
        counts = dynamic_state["session_counts"]
        total = dynamic_state["total_predictions"]
        history = dynamic_state["history"]

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
            "model_loaded": DYNAMIC_MODEL_LOADED
        })

@app.route('/static/api/analytics')
def static_get_analytics():
    with static_lock:
        hist, counts, total = static_state["history"], static_state["counts"], static_state["total"]
        avg_conf = (sum(h["conf"] for h in hist) / len(hist)) if hist else 0
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:15]
        return jsonify({
            "total": total, "unique": len(counts), "avg_conf": round(avg_conf, 1),
            "top": [{"letter": l, "count": c, "pct": round(c / max(total, 1) * 100, 1)} for l, c in top],
            "history": hist[:30], "all_counts": counts,
        })

@app.route('/dynamic/api/camera/control', methods=['POST'])
def dynamic_camera_control():
    try:
        data = request.get_json()
        camera_on = data.get('camera_on', False)
        
        with dynamic_lock:
            dynamic_state["camera_active"] = camera_on
            print(f"📷 Dynamic camera: {'ON' if camera_on else 'OFF'}")
        
        return jsonify({
            "success": True,
            "camera_on": camera_on,
            "message": f"Dynamic camera turned {'ON' if camera_on else 'OFF'}"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/static/api/camera/control', methods=['POST'])
def static_camera_control():
    try:
        data = request.get_json()
        camera_on = data.get('camera_on', False)
        
        with static_lock:
            static_state["camera_active"] = camera_on
            print(f"📷 Static camera: {'ON' if camera_on else 'OFF'}")
        
        return jsonify({
            "success": True,
            "camera_on": camera_on,
            "message": f"Static camera turned {'ON' if camera_on else 'OFF'}"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/dynamic/api/clear_history', methods=['POST'])
def dynamic_clear_history():
    try:
        with dynamic_lock:
            dynamic_state["history"] = []
            dynamic_state["session_counts"] = {}
            dynamic_state["total_predictions"] = 0
            dynamic_state["current_prediction"] = "Waiting..."
            dynamic_state["confidence"] = 0.0
            dynamic_state["all_probs"] = {}
        
        print("🗑️ Dynamic history cleared")
        return jsonify({"success": True, "message": "Dynamic history cleared"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/static/api/clear_history', methods=['POST'])
def static_clear_history():
    try:
        with static_lock:
            static_state["history"] = []
            static_state["counts"] = {}
            static_state["total"] = 0
        
        print("🗑️ Static history cleared")
        return jsonify({"success": True, "message": "Static history cleared"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "dynamic_model_loaded": DYNAMIC_MODEL_LOADED,
        "static_model_loaded": STATIC_MODEL_LOADED,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "dynamic_camera_active": dynamic_state["camera_active"],
        "static_camera_active": static_state["camera_active"],
    }), 200

# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("=" * 60)
    print("🚀 RTSLR Unified Server Starting...")
    print(f"📊 Dynamic Model: {DYNAMIC_MODEL_LOADED} | Static Model: {STATIC_MODEL_LOADED}")
    print(f"🎥 MediaPipe: {MEDIAPIPE_AVAILABLE}")
    print(f"💻 Device: {device}")
    print(f"🎯 Dynamic Classes: {len(DYNAMIC_CLASSES)} | Static Classes: {len(STATIC_CLASSES)}")
    print(f"🔌 Port: {port}")
    print("=" * 60)
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port, threaded=True)