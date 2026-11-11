import streamlit as st
from src.agent import get_recommendations

st.title("Generative AI Agent for Environmental CO₂ Reduction")

st.markdown("Enter a query about your daily activities to get personalized CO₂ reduction recommendations.")

query = st.text_input("Your query (e.g., 'I drive 20 km daily using a petrol car. How can I reduce my CO₂ emissions?')")

if st.button("Get Recommendations"):
    if query:
        with st.spinner("Generating recommendations..."):
            response = get_recommendations(query)
        st.success("Recommendations:")
        st.markdown(response)
    else:
        st.error("Please enter a query.")
