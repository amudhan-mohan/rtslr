# RTSLR - Indian Sign Language Recognition System

## 📋 Overview

RTSLR is a real-time Indian Sign Language (ISL) recognition system powered by Deep Learning. The system uses an LSTM-based neural network to recognize 49 different ISL gestures from live webcam feed, making sign language accessible to everyone.

## 🎯 Project Highlights

- **49 ISL Gesture Classes** - Comprehensive vocabulary covering greetings, actions, questions, and more
- **Real-time Recognition** - 30fps processing with minimal latency (~33ms on GPU)
- **High Accuracy** - 98.48% test accuracy with 0.98 Macro F1 Score
- **Multi-Platform** - Runs on CPU/GPU with automatic device detection
- **Responsive UI** - Mobile-friendly interface with dark theme

## 🏗️ Architecture

### Model Architecture
- **Input**: 258 keypoints per frame (33 pose × 4 + 21 hand × 3 × 2)
- **Sequence Length**: 30 frames
- **LSTM Layers**: 2 layers with 128 hidden units
- **Dropout**: 0.5 for regularization
- **Output**: 49 gesture classes with softmax activation

### Technology Stack

| Component | Technologies |
|-----------|-------------|
| **Frontend** | HTML5, Tailwind CSS, Chart.js |
| **Backend** | Flask (Python) |
| **ML Framework** | PyTorch |
| **Computer Vision** | MediaPipe, OpenCV |
| **Data Processing** | NumPy |

## 📁 Project Structure

<pre>
rtslr/
├── templates/
│   ├── index.html 
│   ├── inference.html
│   ├── analytics.html 
│   └── data_analytics.html
├── app.py
├── sign_lstm_best.pt
└── requirements.txt
</pre>


## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Webcam for live recognition
- CUDA-capable GPU (optional, for faster inference)

### Step 1: Clone/Download the Project
```bash
git clone https://github.com/amudhan-mohan/rtslr.git
cd rtslr
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python app.py
```

### Step 5: Access the Application

Open your browser and navigate to:
```bash
http://localhost:5000
```

## 📊 Dataset Information

### Dataset Composition

| Split | Samples | Percentage |
|-------|---------|------------|
| **Training** | 2,623 | 80% |
| **Testing** | 656 | 20% |
| **Total** | **3,279** | **100%** |

## 🎯 Gesture Categories

| Category | Count | Examples |
|----------|-------|----------|
| 👋 Greetings & Basics | 12 | hello, bye, thank_you, welcome, sorry, good |
| 👤 People & Identity | 10 | i/me, you, my, your, name, who, teacher |
| ⚡ Actions & Verbs | 11 | do, write, read, study, look, come, help |
| 📍 Place, Time & Questions | 16 | home, work, class, here, now, what, how |

## 🎮 Usage Guide

### Live Recognition (Inference Page)

1. Click on "Inference" in the navigation bar
2. Allow camera access when prompted
3. Click "Turn On" to start the camera
4. Perform ISL gestures in front of the camera
5. View real-time predictions with confidence scores

### Features

- **🔍 Real-time Landmark Detection:** Skeletal and hand landmarks overlaid on video
- **📊 Confidence Scoring:** Percentage-based confidence for each prediction
- **💬 Sentence Builder:** Auto-builds sentences from detected gestures
- **📈 Session Analytics:** Track recognition history and statistics
- **🗂️ Dataset Analytics:** View comprehensive dataset distribution

## 📈 Performance Metrics

### Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | **98.48%** |
| Macro Precision | 0.99 |
| Macro Recall | 0.98 |
| Macro F1 Score | 0.98 |
| Weighted F1 Score | 0.98 |

---

### Per-Category Performance

| Category | Accuracy |
|----------|----------|
| Greetings | 94% |
| Identity | 91% |
| Actions | 88% |
| Questions | 86% |
| Complex Gestures | 82% |

---

### Inference Speed

| Device | Speed |
|--------|-------|
| GPU | ~33ms per frame |
| CPU | ~80-100ms per frame |

---

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/instance` | GET | Live recognition interface |
| `/analytics` | GET | Session analytics page |
| `/video_feed` | GET | Video stream endpoint |
| `/api/state` | GET | Current recognition state |
| `/api/analytics` | GET | Session analytics data |
| `/api/camera/control` | POST | Camera on/off control |
| `/api/clear_history` | POST | Clear recognition history |

## 🛠️ Technologies Used

### Backend

- 🐍 **Flask** - Web framework
- 🔥 **PyTorch** - Deep learning framework
- ✋ **MediaPipe** - Landmark detection
- 👁️ **OpenCV** - Video processing
- 🔢 **NumPy** - Numerical operations

### Frontend

- 🎨 **Tailwind CSS** - Styling
- 📊 **Chart.js** - Data visualization
- 🖼️ **Font Awesome** - Icons
- ✍️ **Google Fonts** - Typography (Syne, JetBrains Mono)

## 📝 Model Training Details

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_size` | 258 | Keypoint dimensions |
| `hidden_size` | 128 | LSTM hidden units |
| `num_layers` | 2 | LSTM layers |
| `dropout` | 0.5 | Dropout rate |
| `sequence_length` | 30 | Frames per sequence |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 0.001 | Adam optimizer |
| `epochs` | 100 | Total epochs |
| `best_epoch` | 62 | Best checkpoint |

## 🔄 Data Processing Pipeline

| Step | Description |
|------|-------------|
| **Frame Capture** | Webcam feed at 30 FPS |
| **Landmark Extraction** | MediaPipe extracts 258 keypoints |
| **Normalization** | Spatial normalization relative to hip center |
| **EMA Smoothing** | Exponential Moving Average (α=0.5) |
| **Sequence Buffer** | 30-frame sliding window |
| **LSTM Inference** | Predict gesture class |
| **Voting Buffer** | 20-frame majority voting (12/20 threshold) |

---

## 🎨 UI Features

| Feature | Description |
|---------|-------------|
| 🌙 **Dark Theme** | Eye-friendly dark interface |
| 📱 **Responsive Design** | Works on desktop, tablet, and mobile |
| ☰ **Mobile Menu** | Hamburger menu for smaller screens |
| ⚡ **Real-time Updates** | Live confidence scores and predictions |
| 📊 **Interactive Charts** | Dynamic data visualization with Chart.js |

---

## 📄 License

This project is developed for academic purposes as part of the Bachelor of Engineering program.

---

## 👥 Team

| Role | Name/Details |
|------|--------------|
| **Project Guide** | Dr. R. PRIYA, Professor, Annamalai University |
| **Team Members Name** | AADHITHYA S, AMUDHAN M, DHARUN R S, KARTHIKEYAN V|
| **Team Members Course with Year** | B.E. Computer Science and Engineering, Final Year |

---

## 🙏 Acknowledgments

- Indian Sign Language Research and Training Centre (ISLRTC)
- MediaPipe team for pose and hand landmark detection
- PyTorch community for deep learning frameworks

---

## 📞 Support

For issues or queries:

- Check the console logs for errors
- Ensure webcam is properly connected
- Verify all dependencies are installed correctly
- Make sure the model file `sign_lstm_best.pt` is in the root directory

---

## 🔄 Future Enhancements
- Add more gesture classes (100+)
- Implement continuous sentence recognition
- Add speech-to-text output for recognized signs
- Deploy as web application with cloud backend
- Add user authentication and progress tracking
- Implement transfer learning for new users
- Add support for regional ISL variations

---

<div align="center">
  
**© 2025 RTSLR** — Final Year Project. Built with ❤️ for the deaf community of India.

</div>