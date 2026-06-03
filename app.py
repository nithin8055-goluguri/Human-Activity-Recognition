import streamlit as st
import cv2
import numpy as np
import pickle
import mediapipe as mp

st.set_page_config(page_title="AI Human Activity Dashboard", layout="wide")

st.title("🏋️‍♂️ Multi-Action Human Activity Recognition Website")
st.markdown("Toggle the checkbox below to initialize your laptop camera and recognize your action live.")

# ─── LOAD TRAINED COMPONENTS SAFELY ───
model, le, scaler = None, None, None
try:
    model  = pickle.load(open("model.pkl",         "rb"))
    le     = pickle.load(open("label_encoder.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl",        "rb"))
    st.sidebar.success(f"🤖 Connected to Dataset! Ready to classify {len(le.classes_)} actions.")
except Exception:
    st.sidebar.error("⚠️ Model pipeline not found. Run train_model.py first.")

# ─── FRONTEND METRIC INPUTS ───
st.sidebar.header("Activity Parameters")
weight_kg = st.sidebar.number_input("Weight (kg):", min_value=30, max_value=150, value=70)
duration_min = st.sidebar.slider("Duration (mins):", 1, 60, 15)

CALORIE_RATES = {
    "SITTING": 1.3, "STANDING": 1.5, "WALKING": 4.5, "WAVING": 2.5,
    "SQUATS": 5.5, "BICEP_CURLS": 3.5, "JUMPING_JACKS": 8.0, 
    "RUNNING": 10.0, "LAYING": 1.0, "PUSHUPS": 7.0, "PUNCHING": 6.5
}
FITNESS_TIPS = {
    "SITTING": "Take a 2-minute stretch break to loosen up your spine.",
    "STANDING": "Great posture! Shift your weight between feet occasionally.",
    "WALKING": "Fantastic low-impact cardio. Keep moving!",
    "WAVING": "Arm mobilization active. Keep moving your shoulders.",
    "SQUATS": "Excellent for leg drive! Keep your heels glued to the floor.",
    "BICEP_CURLS": "Focus on controlled movements. Squeeze your arms at the top.",
    "JUMPING_JACKS": "High-intensity fat burner! Landing softly protects your joints.",
    "RUNNING": "Amazing cardio output! Keep your breathing deep and rhythmic.",
    "LAYING": "Muscle recovery active. Make sure to get plenty of rest.",
    "PUSHUPS": "Keep your core braced tight and lower your chest evenly.",
    "PUNCHING": "Incredible rotational strength output! Keep your guard up."
}

# Initialize MediaPipe Engine
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# UI Trigger Checkbox for local camera loop
run_app = st.checkbox("Toggle to Turn On Laptop Camera")
FRAME_WINDOW = st.image([])

if run_app:
    # Open local laptop webcam frame directly (bypasses browser server errors)
    cap = cv2.VideoCapture(0)
    
    while run_app:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to fetch camera frame.")
            break
            
        # Transform frame color coordinates
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_img)
        
        action_label = "Analyzing Position..."
        calories = 0.0
        tip = "Ensure your full body is visible in the laptop camera."

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            pose_row = []
            for lm in landmarks:
                pose_row.extend([lm.x, lm.y, lm.z, lm.visibility])
            
            if model is not None:
                try:
                    sample = np.array(pose_row).reshape(1, -1)
                    sample_scaled = scaler.transform(sample)
                    pred = model.predict(sample_scaled)
                    
                    # Safely extract string label from prediction output format
                    raw_pred = le.inverse_transform(pred)[0] if isinstance(pred, np.ndarray) else le.inverse_transform([pred])[0]
                    action_label = str(raw_pred).upper()
                    
                    # Compute fitness metrics
                    rate = CALORIE_RATES.get(action_label, 2.0)
                    calories = round(rate * (weight_kg / 70.0) * duration_min, 2)
                    tip = FITNESS_TIPS.get(action_label, "Keep active!")
                except Exception as e:
                    pass

            # Overlay visual skeletal components on image canvas
            mp_drawing.draw_landmarks(rgb_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Draw real-time UI data overlay box
            cv2.rectangle(rgb_img, (0, 0), (640, 115), (20, 20, 20), -1)
            cv2.putText(rgb_img, f"ACTION: {action_label}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(rgb_img, f"ENERGY: {calories} kcal", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)
            cv2.putText(rgb_img, f"TIP: {tip}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

        # Update the live canvas matrix item inside the browser
        FRAME_WINDOW.image(rgb_img)
        
    cap.release()
else:
    st.info("Camera is currently disabled.")
