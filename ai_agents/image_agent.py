import streamlit as st
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
from crewai_tools import WebsiteSearchTool, SerperDevTool
import os

# Load environment variables
load_dotenv()

# Configure the Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# Define the model
model = genai.GenerativeModel("gemini-1.5-pro")
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key="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")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate content based on the image
    prompt = "Analyze the image and list the products present in the image."
    response = model.generate_content([prompt, image])

    # Agent for analyzing the image and providing product recommendations
    image_analysis_agent = Agent(
        role="Analysis and Recommendation Expert",
        goal=f"""Analyze an {response} to answer questions and provide product recommendations.""",
        tool=[WebsiteSearchTool, SerperDevTool],
        verbose=True,
        backstory="""You are an expert at analyzing and extract relevant information.
            Your task is to help users understand the content and suggest the best products available online.""",
        llm=llm,
        allow_delegation=True
    )

    # Get user questions about the image
    questions = st.text_input("Enter your questions about the image (separated by commas):")

    if questions:
        # Create the task for image analysis
        analyze_image_task = Task(
            description=(
                f"Answer any user questions ({questions}) based on the image analysis, and provide recommendations "
                "for products similar to those asked. Use relevant websites to gather "
                "information about these products. Give links to relevant websites where the product can be purchased "
                "and their approximate prices."
            ),
            expected_output="Detailed answers to user questions and a list of recommendations for products with links to where they can be purchased.",
            tool=[WebsiteSearchTool, SerperDevTool],
            agent=image_analysis_agent,
        )

        # Create the crew to handle the task
        image_analysis_crew = Crew(
            agents=[image_analysis_agent],
            tasks=[analyze_image_task],
            process=Process.sequential
        )

        # Prepare the crew and execute the analysis
        result = image_analysis_crew.kickoff()
        st.write(result)
