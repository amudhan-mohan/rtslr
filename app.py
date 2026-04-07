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
import os
import signal
import sys

app = Flask(__name__)

# ==========================================
# Model Architecture
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
CONFIDENCE_THRESHOLD = 0.85
VOTING_THRESHOLD = 12
BUFFER_SIZE = 20

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
camera_retry_count = 0
MAX_CAMERA_RETRIES = 3

# ==========================================
# Load Model with better error handling
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_LOADED = False
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = SignLanguageLSTM(input_size=KEYPOINT_DIM, num_classes=len(CLASSES)).to(device)
        # Use weights_only=True for security in production
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        MODEL_LOADED = True
        print(f"✅ Model loaded successfully on {device}")
        print(f"📊 Model classes: {len(CLASSES)} gesture classes")
    except Exception as e:
        print(f"⚠️ Model not loaded: {e}. Running in demo mode.")
        MODEL_LOADED = False
else:
    print(f"⚠️ Model file not found at {MODEL_PATH}. Running in demo mode.")
    MODEL_LOADED = False

# MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# Helper Functions
# ==========================================
def extract_keypoints(pose_results, hand_results):
    """Extract pose and hand keypoints from MediaPipe results"""
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
    """Normalize keypoints relative to hip center and shoulder distance"""
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
    """Apply Exponential Moving Average for smoothing"""
    if prev_lms is None:
        return current_lms
    return current_lms * alpha + prev_lms * (1 - alpha)

def graceful_shutdown(signum, frame):
    """Handle graceful shutdown"""
    print("\n🛑 Shutting down gracefully...")
    with state_lock:
        state["running"] = False
    global cap
    if cap is not None:
        cap.release()
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

# ==========================================
# Video Generator (Enhanced Production Version)
# ==========================================
def generate_frames():
    global cap, camera_retry_count
    
    sequence = deque(maxlen=SEQUENCE_LENGTH)
    prediction_buffer = deque(maxlen=BUFFER_SIZE)
    previous_landmarks = None
    frame_count = 0
    fps_update_interval = 30
    start_time = time.time()
    
    # For demo mode when model not loaded
    demo_mode = not MODEL_LOADED
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        
        while state["running"]:
            try:
                # Check camera state
                with state_lock:
                    camera_active = state["camera_active"]
                
                # Handle camera initialization when turned ON
                if camera_active and cap is None:
                    with cap_lock:
                        cap = cv2.VideoCapture(0)
                        if not cap.isOpened():
                            camera_retry_count += 1
                            print(f"❌ Failed to open camera (attempt {camera_retry_count}/{MAX_CAMERA_RETRIES})")
                            if camera_retry_count >= MAX_CAMERA_RETRIES:
                                with state_lock:
                                    state["camera_active"] = False
                                    camera_active = False
                                print("⚠️ Max camera retries reached. Please check your camera connection.")
                            else:
                                time.sleep(1)
                            continue
                        else:
                            print("✅ Camera started successfully")
                            camera_retry_count = 0
                
                # Handle camera release when turned OFF
                if not camera_active and cap is not None:
                    with cap_lock:
                        cap.release()
                        cap = None
                        print("📷 Camera stopped")
                        sequence.clear()
                        prediction_buffer.clear()
                        previous_landmarks = None
                
                # If camera is OFF or not available, show placeholder
                if not camera_active or cap is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "CAMERA OFF", (220, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 106, 130), 2)
                    cv2.putText(frame, "Click 'Turn On' to start", (200, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 106, 130), 1)
                    
                    # Add demo mode indicator
                    if demo_mode:
                        cv2.putText(frame, "DEMO MODE - Model not loaded", (180, 320), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1)
                    
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
                    print("⚠️ Failed to read frame from camera")
                    time.sleep(0.01)
                    continue
                
                # Calculate FPS
                frame_count += 1
                if frame_count % fps_update_interval == 0:
                    elapsed_time = time.time() - start_time
                    fps = fps_update_interval / elapsed_time if elapsed_time > 0 else 0
                    print(f"📊 FPS: {fps:.2f}")
                    start_time = time.time()
                
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
                
                # Only do inference if model is loaded
                if not demo_mode:
                    try:
                        # Keypoint pipeline
                        keypoints = extract_keypoints(pose_res, hand_res)
                        normalized_keypoints = normalize_frame(keypoints)
                        smoothed_keypoints = apply_ema(normalized_keypoints, previous_landmarks, EMA_ALPHA)
                        previous_landmarks = smoothed_keypoints
                        sequence.append(smoothed_keypoints)
                        
                        local_prediction = "Waiting..."
                        local_confidence = 0.0
                        all_probs = {}
                        
                        if len(sequence) == SEQUENCE_LENGTH:
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
                                
                                if local_confidence > CONFIDENCE_THRESHOLD:
                                    local_prediction = CLASSES[max_idx]
                                
                                prediction_buffer.append(local_prediction)
                                
                                if len(prediction_buffer) > 0:
                                    vote_tally = Counter(prediction_buffer).most_common(1)[0]
                                    winning_gesture, winning_votes = vote_tally
                                    if winning_votes >= VOTING_THRESHOLD:
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
                                                print(f"🎯 Gesture detected: {winning_gesture} ({local_confidence*100:.1f}%)")
                    except Exception as e:
                        print(f"⚠️ Inference error: {e}")
                        continue
                
                # Encode and yield frame
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
            except Exception as e:
                print(f"❌ Frame generation error: {e}")
                continue
    
    # Cleanup
    if cap is not None:
        cap.release()
    print("🛑 Video generator stopped")

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
            "model_loaded": MODEL_LOADED
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
            "model_loaded": MODEL_LOADED
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
        
        print("🗑️ History cleared")
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

@app.route('/api/model/info')
def model_info():
    """Get model information"""
    return jsonify({
        "model_loaded": MODEL_LOADED,
        "device": str(device),
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "sequence_length": SEQUENCE_LENGTH,
        "keypoint_dim": KEYPOINT_DIM,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "voting_threshold": VOTING_THRESHOLD
    })

@app.route('/health')
def health_check():
    """Health check endpoint for Render.com"""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "camera_active": state["camera_active"],
        "total_predictions": state["total_predictions"]
    }), 200

# ==========================================
# Error Handlers
# ==========================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ==========================================
# Production Entry Point
# ==========================================
if __name__ == '__main__':
    # Get port from environment variable (for Render.com)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("=" * 50)
    print("🚀 RTSLR Server Starting...")
    print(f"📊 Model Loaded: {MODEL_LOADED}")
    print(f"💻 Device: {device}")
    print(f"🎯 Classes: {len(CLASSES)}")
    print(f"🔌 Port: {port}")
    print(f"🐛 Debug Mode: {debug_mode}")
    print("=" * 50)
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port, threaded=True)