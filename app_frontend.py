import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Car Damage Detection", layout="wide")

st.title("🚗 Car Damage Detection System")
st.markdown("Upload a car image to detect damage using CNN")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This application uses a Convolutional Neural Network (CNN) "
    "to detect car damage from images."
)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose a car image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("Prediction Results")
    
    if uploaded_file:
        if st.button("🔍 Analyze Damage", key="analyze"):
            with st.spinner("Processing..."):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post("http://localhost:8000/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        damage_class = result['class']
                        confidence = result['confidence']
                        predictions = result['predictions']
                        
                        st.success("Analysis Complete!")
                        
                        # Metrics
                        col_metric1, col_metric2 = st.columns(2)
                        with col_metric1:
                            st.metric("Status", damage_class)
                        with col_metric2:
                            st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Detailed probabilities
                        st.subheader("Detailed Probabilities")
                        st.progress(predictions['damaged'], text=f"Damaged: {predictions['damaged']:.2%}")
                        st.progress(predictions['not_damaged'], text=f"Not Damaged: {predictions['not_damaged']:.2%}")
                    else:
                        st.error(f"Error: {response.text}")
                
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Cannot connect to API. Make sure the backend is running on http://localhost:8000")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & FastAPI")