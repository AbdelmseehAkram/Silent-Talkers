
# SignVision: AI-Powered ASL & ArSL Translator

> Developed by **Silent Talkers**  
> Team Members: Abdulaziz Mohamed, Abdelmseeh Akram, Ehab Gerges, Ahmed Osama, Fares Zakaria  
> Supervised by: Dr. Mohamed Masoud

---

## 📖 Overview

SignVision is an AI-powered system designed to translate American Sign Language (ASL) and Arabic Sign Language (ArSL) hand gestures into readable text (English/Arabic) and audio output.  
The project aims to bridge the communication gap between the Deaf & Mute (D&M) community and the broader society through real-time hand gesture recognition.

---

## 🎯 Objective

- Enable real-time recognition of ASL and ArSL fingerspelling.
- Translate recognized signs into English and Arabic text and generate audio output.
- Provide a user-friendly Human-Computer Interface (HCI) accessible to both signers and non-signers.

---

## 📦 Project Modules

- **Data Acquisition:** Capturing ASL and ArSL gestures using a webcam.
- **Preprocessing & Feature Extraction:** Using MediaPipe for hand landmark detection and conversion to structured vectors.
- **Model Training:**  
  - **CNN**: For image-based gesture recognition.
  - **MLP**: For landmark-based gesture recognition.
- **Real-time Translation:** Outputting recognized gestures as text and speech.
- **User Interface:** Built with Tkinter for easy model interaction.

---

## 🛠️ Requirements

### Hardware:
- Webcam or laptop camera
- GPU (optional for faster inference)

### Software:
- OS: Windows 10+ or Linux
- Python 3.9.5+
- Libraries: TensorFlow, Keras, OpenCV, Mediapipe, NumPy, Scikit-learn, Matplotlib, Seaborn

---

## 🧠 Model Details

- **MLP Network:**
  - Input: 63 features (21 landmarks × (x, y, z))
  - Hidden Layers: 
    - 128 neurons (ReLU activation) + 20% dropout
    - 64 neurons (ReLU activation)
  - Output: 27 classes (A-Z + blank) with Softmax activation
- **Training:**
  - Optimizer: Adam
  - Loss Function: Categorical Crossentropy
  - Batch Size: 32
  - Epochs: 50 (with early stopping)

---

## 📊 Performance

| Model | Accuracy |
|:------|:--------:|
| ASL Recognition | 88.85% |
| ArSL Recognition | 89% |

- Under optimal conditions (good lighting, clean background), accuracy can reach up to **99%**.
- Evaluation metrics: Precision, Recall, F1-score, Confusion Matrix, Training/Validation Loss.

---

## 🖥️ User Interface

- Built using **Tkinter**.
- Features:
  - Live webcam feed.
  - Model switching (ASL/ArSL).
  - Real-time sign prediction with confidence.
  - Sentence building and editing (space, delete).
  - Multilingual support (English/Arabic).

---

## 🚀 Future Work

- Developing a cross-platform **Android application** for mobile real-time ASL/ArSL translation.
- Further improvement of gesture differentiation for visually similar signs.

---

## 📚 References

- [MediaPipe Hands](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)
- [TensorFlow Feature Columns](https://www.tensorflow.org/tutorials/structured_data/feature_columns)
- [OpenCV Documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Scikit-learn Train/Test Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [MLP Deep Learning Book](https://www.deeplearningbook.org/contents/mlp.html)
- [Sign Language Translator App (GitHub Repo)](https://github.com/AbdelmseehAkram/sign-language-translator-app)
- [Demo UI (Huggingface)](https://huggingface.co/spaces/ahmedos13/SignLanguage)

---

> Made with ❤️ by Silent Talkers
