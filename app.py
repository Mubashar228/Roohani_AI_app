import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
from deepface import DeepFace
import random

# --- Streamlit App UI ---
st.set_page_config(page_title="Roohaniyat AI", layout="centered")
st.title("🔮 Roohaniyat AI – Chehra aur Haath se Kismat ka Haal")
st.markdown("Upload apna **haath ya chehra ka image** aur paayein roohani predictions ✨")

# --- File Upload ---
uploaded_file = st.file_uploader("📷 Image upload karein", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save the uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Read the image using OpenCV
    img = cv2.imread(tfile.name)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    # --- Face Detection using DeepFace ---
    try:
        result = DeepFace.analyze(img_path=tfile.name, actions=['emotion', 'age', 'gender'], enforce_detection=False)[0]

        st.subheader("🧠 Chehra ka Analysis:")
        st.write(f"Emotion: `{result['dominant_emotion']}`")
        st.write(f"Age: `{result['age']}`")
        st.write(f"Gender: `{result['gender']}`")

        # Basic Personality Prediction
        emotion = result['dominant_emotion']
        if emotion in ['happy', 'surprise']:
            st.success("✨ Aap ek positive shakhs hain, jinki rozi mein barkat aur pyar barhne ka imkaan hai.")
        elif emotion in ['sad', 'fear', 'angry']:
            st.warning("⚠️ Aap kuch emotional challenges se guzar rahe hain, lekin roohani madad aapka intezar kar rahi hai.")
        else:
            st.info("🔍 Aap sochne wale shakhs hain – agla saal aapki zindagi mein naya mod laa sakta hai.")

    except Exception as e:
        st.error("Chehra detect nahi ho saka. Kya aap clear image upload kar rahe hain?")

    # --- Palm (Hand) Detection using Mediapipe ---
    st.subheader("✋ Haath ka Analysis:")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        st.success("✅ Haath detect ho gaya hai!")

        # Random spiritual message (for now)
        messages = [
            "🕊️ Aapka naseeb roshan hai – naya safar jald shuru hoga.",
            "💍 Shadi ke imkaan agle 1-2 saal mein hain.",
            "👶 Bachon ki khushkhabri aapki zindagi mein barkat laayegi.",
            "📈 Career mein tarraqqi ka waqt qareeb hai.",
            "🙏 Roohani tor par aap mazboot hain – duaon ka asar ho raha hai."
        ]
        st.info(random.choice(messages))
    else:
        st.error("Haath clearly detect nahi ho saka. Kya aap ne clear haath ka photo diya hai?")

