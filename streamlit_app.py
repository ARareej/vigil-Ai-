import streamlit as st
import os
import sys
import requests
from datetime import datetime
from PIL import Image
from io import BytesIO

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_prompt

def main():
    # --------------------------
    # Custom CSS
    # --------------------------
    st.set_page_config(page_title="Vigil AI - AI Safety Layer", page_icon="🛡️", layout="centered", initial_sidebar_state="collapsed")

    st.markdown("""
        <style>
        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #0d1433 50%, #0a0e27 100%);
            background-size: 200% 200%;
            animation: gradientShift 15s ease infinite;
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Dotted background pattern */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: radial-gradient(circle, rgba(45, 212, 191, 0.1) 1px, transparent 1px);
            background-size: 20px 20px;
            pointer-events: none;
            z-index: 0;
        }

        /* Glassmorphism cards */
        .glass-card {
            background: rgba(20, 30, 60, 0.75);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(70, 190, 180, 0.25);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
            position: relative;
            z-index: 1;
        }

        .glass-card-success {
            background: rgba(21, 128, 61, 0.25);
            border: 1px solid rgba(74, 222, 128, 0.4);
            box-shadow: 0 10px 40px rgba(21, 128, 61, 0.3);
        }

        .glass-card-warning {
            background: rgba(185, 28, 28, 0.25);
            border: 1px solid rgba(248, 113, 113, 0.4);
            box-shadow: 0 10px 40px rgba(185, 28, 28, 0.3);
        }

        /* Button styling */
        .stButton>button {
            background: linear-gradient(135deg, #2dd4bf 0%, #0ea5e9 100%);
            color: white;
            border: none;
            border-radius: 16px;
            padding: 0.875rem 2rem;
            font-size: 1.125rem;
            font-weight: 700;
            box-shadow: 0 6px 24px rgba(45, 212, 191, 0.4);
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            width: 100%;
        }

        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 32px rgba(45, 212, 191, 0.55);
        }

        /* Input styling */
        .stTextArea>div>div>textarea {
            background: rgba(15, 25, 50, 0.85);
            border: 1px solid rgba(70, 190, 180, 0.3);
            color: white;
            border-radius: 16px;
            font-size: 1.05rem;
            padding: 1.25rem;
            line-height: 1.6;
        }

        /* Title styling */
        .main-title {
            font-size: 3.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #2dd4bf 0%, #86efac 50%, #22d3ee 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-align: center;
        }

        .subtitle {
            color: #cbd5e1;
            font-size: 1.25rem;
            margin-bottom: 2rem;
            text-align: center;
        }

        /* Feature badges */
        .feature-badge {
            background: rgba(45, 212, 191, 0.15);
            border: 1px solid rgba(45, 212, 191, 0.3);
            padding: 0.5rem 1.25rem;
            border-radius: 50px;
            color: #a5f3fc;
            font-weight: 600;
            font-size: 0.9rem;
        }

        /* Label badges */
        .safe-badge {
            background: linear-gradient(135deg, #22c55e 0%, #4ade80 100%);
            color: white;
            padding: 0.75rem 2rem;
            border-radius: 50px;
            font-size: 1.5rem;
            font-weight: 800;
            display: inline-block;
            margin-bottom: 0.5rem;
            box-shadow: 0 4px 20px rgba(34, 197, 94, 0.4);
        }

        .blocked-badge {
            background: linear-gradient(135deg, #ef4444 0%, #fb7185 100%);
            color: white;
            padding: 0.75rem 2rem;
            border-radius: 50px;
            font-size: 1.5rem;
            font-weight: 800;
            display: inline-block;
            margin-bottom: 0.5rem;
            box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #64748b;
            margin-top: 3.5rem;
            padding-top: 2.5rem;
            border-top: 1px solid rgba(100, 116, 139, 0.25);
            font-size: 1rem;
        }

        /* Section headers */
        .section-title {
            color: #e2e8f0;
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # --------------------------
    # Hero Section
    # --------------------------
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="main-title">🛡️ Vigil AI - AI Safety Layer</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Pre-generation guardrail for image generation systems</div>', unsafe_allow_html=True)

        # Feature badges
        st.markdown("""
            <div style="display: flex; flex-wrap: wrap; gap: 0.75rem; justify-content: center; margin-bottom: 2.5rem;">
                <div class="feature-badge">✓ Prompt Classification</div>
                <div class="feature-badge">✓ Policy Enforcement</div>
                <div class="feature-badge">✓ Jailbreak Detection</div>
                <div class="feature-badge">✓ Safe Image Generation</div>
            </div>
        """, unsafe_allow_html=True)

        # --------------------------
        # Prompt Section (without glass-card wrapper)
        # --------------------------
        st.markdown('<div class="section-title"> Image Prompt</div>', unsafe_allow_html=True)

        prompt = st.text_area(
            "",
            placeholder="Describe the image you want to generate…",
            height=140,
            label_visibility="collapsed"
        )

        # Check Safety Button
        if st.button("🔍 Check Safety", type="primary"):
            if not prompt.strip():
                st.warning("Please enter a prompt first.")
            else:
                try:
                    with st.spinner("Analyzing prompt safety..."):
                        prediction = predict_prompt(prompt)

                    pred_label = prediction["predicted_label"]
                    confidence = prediction["confidence"]

                    st.session_state["prediction"] = prediction
                    st.session_state["pred_label"] = pred_label
                    st.session_state["confidence"] = confidence
                    st.session_state["show_result"] = True

                except Exception as e:
                    st.error("Error analyzing prompt. Please try again.")
                    st.session_state["show_result"] = False

        # --------------------------
        # Results Section
        # --------------------------
        if "show_result" in st.session_state and st.session_state["show_result"]:
            st.divider()
            prediction = st.session_state["prediction"]
            pred_label = st.session_state["pred_label"]
            confidence = st.session_state["confidence"]

            if pred_label == "SAFE":
                st.markdown('<div class="glass-card glass-card-success">', unsafe_allow_html=True)
                st.markdown(f'<div class="safe-badge">✅ SAFE</div>', unsafe_allow_html=True)
                st.subheader("Prompt approved", anchor=False)

                col_a, col_b = st.columns(2)
                col_a.metric("Label", "SAFE")
                col_b.metric("Confidence", f"{confidence:.1%}")

                if st.button("✨ Generate Image", key="gen_button"):
                    try:
                        with st.spinner("Generating your image..."):
                            # Call Pollinations AI
                            pollinations_url = f"https://image.pollinations.ai/prompt/{prompt.replace(' ', '%20')}"
                            img_response = requests.get(pollinations_url)
                            img = Image.open(BytesIO(img_response.content))

                            # Save image
                            images_dir = os.path.join("outputs", "generated_images")
                            os.makedirs(images_dir, exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_path = os.path.join(images_dir, f"generated_{timestamp}.png")
                            img.save(img_path)

                            # Display image
                            st.image(img, caption="Generated Successfully", use_container_width=True)
                            st.success(f"Image saved to {img_path}")

                    except Exception as e:
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('<div class="glass-card glass-card-warning">', unsafe_allow_html=True)
                        st.error("⚠️ Image generation failed. Please try again later.")
                        st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.markdown('<div class="glass-card glass-card-warning">', unsafe_allow_html=True)
                st.markdown(f'<div class="blocked-badge">❌ BLOCKED</div>', unsafe_allow_html=True)

                col_a, col_b = st.columns(2)
                col_a.metric("Label", pred_label)
                col_b.metric("Confidence", f"{confidence:.1%}")

                st.warning("Prompt violates safety policy and cannot be sent to the image generator.")

                if prediction["was_translated"]:
                    st.info(f"Translated prompt: {prediction['processed_prompt']}")

                st.markdown('</div>', unsafe_allow_html=True)

        # --------------------------
        # Footer
        # --------------------------
        st.markdown("""
            <div class="footer">
                <p>🔒 Vigil AI - Advanced Safety Guardrail for Generative AI Systems</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
