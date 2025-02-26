import os
from dotenv import load_dotenv  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
import streamlit as st  # type: ignore
from langchain.chains import LLMChain  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore
import google.generativeai as genai  # type: ignore
import pandas as pd  # type: ignore
import io
import plotly.graph_objects as go  # type: ignore

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not found in .env file.")
    st.stop()

# Verify Gemini Flash availability
genai.configure(api_key=gemini_api_key)
available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

if 'models/gemini-2.0-flash-exp' not in available_models:
    st.error("❌ gemini-2.0-flash-exp is not available for your API key. Please check your Google Cloud project and API key.")
    st.stop()

# Initialize Google GenAI model (Gemini Flash)
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=gemini_api_key)
except Exception as e:
    st.error(f"❌ Error initializing Gemini Flash model: {e}")
    st.stop()

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["source", "destination"],
    template="""
    You are a travel planning assistant. Provide travel options from {source} to {destination}. 
    Present the information in a structured table format with the following columns:

    | Travel Type | Price (Estimated) | Time (Estimated) | Description | Comfort Level (1-5, 5 being highest) | Directness (Direct/Indirect) |
    |-------------------|-------------------|-------------------|-------------|------------------------------------|-----------------------------|
    | Cab/Taxi          |                   |                   |             |                                    |                             |
    | Train             |                   |                   |             |                                    |                             |
    | Bus               |                   |                   |             |                                    |                             |
    | Flight            |                   |                   |             |                                    |                             |
    | Ola/Uber          |                   |                   |             |                                    |                             |

    Fill in the table with estimated prices, travel times, descriptions, comfort levels (1-5), and directness.
    If a mode of transport is unavailable, indicate it in the table.
    """
)

# LangChain LLMChain
travel_chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to generate travel recommendations
def get_travel_recommendations(source, destination):
    try:
        response = travel_chain.run({"source": source, "destination": destination})
        if isinstance(response, str):
            return response
        else:
            return response["text"]
    except Exception as e:
        return f"❌ An error occurred: {e}"

# Streamlit UI
st.title("✈️ AI-Powered Travel Planner")
st.write("Find your optimal travel options! 🗺️")

source = st.text_input("📍 Enter Source City:")
destination = st.text_input("🏁 Enter Destination City:")

if st.button("🚀 Get Travel Options"):
    if source and destination:
        st.write(f"🔍 Generating travel options from {source} to {destination}...")
        recommendations = get_travel_recommendations(source, destination)
        st.write("### 🗂️ Travel Recommendations:")
        st.write(recommendations)
        # CSV Download and Chart Generation
        try:
            table_data = recommendations.strip().split('\n')[2:-1]
            rows = [row.strip().split('|')[1:-1] for row in table_data]
            df = pd.DataFrame(rows, columns=["Travel Type", "Price (Estimated)", "Time (Estimated)", "Description", "Comfort Level", "Directness"])

            # Convert Price and Time to numeric, handling potential errors
            df["Price (Estimated)"] = pd.to_numeric(df["Price (Estimated)"].str.replace(r'[^\d\.]+', '', regex=True), errors='coerce')
            df["Time (Estimated)"] = pd.to_numeric(df["Time (Estimated)"].str.replace(r'[^\d\.]+', '', regex=True), errors='coerce')

            # Create Price Chart
            fig_price = go.Figure([go.Bar(x=df["Travel Type"], y=df["Price (Estimated)"])])
            fig_price.update_layout(title="💰 Price Comparison", xaxis_title="Travel Type", yaxis_title="Price")
            st.plotly_chart(fig_price)

            # Create Time Chart
            fig_time = go.Figure([go.Bar(x=df["Travel Type"], y=df["Time (Estimated)"])])
            fig_time.update_layout(title="⏳ Time Comparison", xaxis_title="Travel Type", yaxis_title="Time")
            st.plotly_chart(fig_time)

            # Create a combined chart
            fig = go.Figure()

            # Add Price line
            fig.add_trace(go.Line(
                x=df["Travel Type"],
                y=df["Price (Estimated)"],
                name="Price",
                marker_color="skyblue"
            ))

            # Add Time line
            fig.add_trace(go.Line(
                x=df["Travel Type"],
                y=df["Time (Estimated)"],
                name="Time",
                marker_color="salmon"
            ))

            # Update layout
            fig.update_layout(
                title="💰⏳ Price and Time Comparison by Travel Type",
                xaxis_title="Travel Type",
                yaxis_title="Value (Price/Time)",
                barmode="group",  # Groups line for each Travel Type
                legend=dict(
                    orientation="h",  # Horizontal legend
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig)

            # CSV Download
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(label="📥 Download Travel Data as CSV", data=csv_buffer.getvalue(), file_name="travel_data.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error processing data or creating charts/CSV: {e}")

    else:
        st.error("❌ Please enter both source and destination cities.")

