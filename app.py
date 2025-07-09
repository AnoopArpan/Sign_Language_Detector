import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import mediapipe as mp
import joblib

# Load trained model and label encoder
model = joblib.load("rf_model.pkl")
le = joblib.load("label_encoder.pkl")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define Streamlit webcam transformer
class HandSignTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])

                if len(landmark_list) == 63:
                    prediction = model.predict([landmark_list])[0]
                    predicted_label = le.inverse_transform([prediction])[0]
                    cv2.putText(img, f"Predicted: {predicted_label}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return img

# Streamlit UI
st.set_page_config(page_title="Sign Language Recognizer", layout="centered")
st.title("🤟 Real-time Sign Language Recognizer")
st.markdown("Show a hand sign in front of your webcam. Press **Q** or stop the stream to exit.")

# Launch webcam streamer
webrtc_streamer(key="sign_stream", video_processor_factory=HandSignTransformer)


