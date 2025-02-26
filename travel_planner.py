import os
import re
import streamlit as st  # type: ignore
from dotenv import load_dotenv  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain.chains import LLMChain  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore
import google.generativeai as genai  # type: ignore

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

# Initialize Google GenAI model
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

    Additionally, provide recommendations for the destination city in a structured format:
    
    Oyo Rooms:
    Oyo Rooms/Hotels Names | Price | Distance | Rating
    Oyo Room |  |  | 
    Oyo Room |  |  | 
    Oyo Room |  |  | 
    Oyo Room |  |  | 
    Oyo Room |  |  | 
    
    Movie Theaters:
    Movie Theaters Names | Showtimes | Distance | Rating
    Movie Theater Name |  |  | 
    Movie Theater Name |  |  | 
    Movie Theater Name |  |  | 
    Movie Theater Name |  |  | 
    Movie Theater Name |  |  | 
    
    Zomato Outlets:
    Zomato Outlets Names | Cuisine | Price Range | Rating
    Zomato Outlet Name |  |  | 
    Zomato Outlet Name |  |  | 
    Zomato Outlet Name |  |  | 
    Zomato Outlet Name |  |  | 
    Zomato Outlet Name |  |  |

    Also, give a brief summary of the destination city, including notable attractions, local cuisine, and any unique cultural experiences.
    """
)

travel_chain = LLMChain(llm=llm, prompt=prompt_template)

def get_travel_recommendations(source, destination):
    try:
        response = travel_chain.run({"source": source, "destination": destination})
        return response if isinstance(response, str) else response["text"]
    except Exception as e:
        return f"❌ An error occurred: {e}"

def extract_section(text, section_name):
    pattern = rf"{section_name}:\s*(\|.*\|[\s\S]*?)(?:\n\n|\Z)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None

def extract_summary(text):
    pattern = r"Also, give a brief summary of the destination city, including notable attractions, local cuisine, and any unique cultural experiences.\s*(.+)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# Streamlit UI
st.title("✈️ AI-Powered Travel Planner")
st.write("Find your optimal travel options and destination recommendations! 🗺️")

source = st.text_input("📍 Enter Source City:")
destination = st.text_input("🏁 Enter Destination City:")

if st.button("🚀 Get Travel Options & Recommendations"):
    if not source or not destination:
        st.error("❌ Please enter both source and destination cities.")
        st.stop()

    st.write(f"🔍 Generating travel options from {source} to {destination}...")
    recommendations = get_travel_recommendations(source, destination)
    st.write("### 🗂️ Travel Recommendations:")
    st.write(recommendations)

