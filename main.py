import streamlit as st
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
from crewai_tools import WebsiteSearchTool, SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def main():
    st.title("Hyperpersonalization and Prompt-based Shopping Experience using Generative AI")

    st.sidebar.header("Upload Files")

    model_type = st.sidebar.selectbox(
        "Choose the type of model",
        ("Recommendation Model", "Image Question Answering Model", "Web Searching Model")
    )

    st.write(f"### Selected Model: {model_type}")

    if model_type == "Image Question Answering Model":
        load_dotenv()

        gemini_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=gemini_api_key)

        model = genai.GenerativeModel("gemini-1.5-pro")
        llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key="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")

        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            prompt = "Analyze the image and list the products present in the image."
            response = model.generate_content([prompt, image])

            # Agent for analyzing the image and providing product recommendations
            image_analysis_agent = Agent(
                role="Analysis and Recommendation Expert",
                goal=f"""Analyze an {response} to answer questions and provide product recommendations.""",
                tool=[WebsiteSearchTool, SerperDevTool],
                verbose=True,
                backstory="""You are an expert at analyzing and extracting relevant information.
                    Your task is to help users understand the content and suggest the best products available online.""",
                llm=llm,
                allow_delegation=True
            )

            questions = st.text_input("Enter your questions about the image (separated by commas):")

            if questions:
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

                image_analysis_crew = Crew(
                    agents=[image_analysis_agent],
                    tasks=[analyze_image_task],
                    process=Process.sequential
                )

                result = image_analysis_crew.kickoff()
                st.write(result)

    elif model_type == "Recommendation Model":
        load_dotenv()

        llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"])

        recommendation_agent = Agent(
            role="Product Recommendation Agent",
            goal="""To give personalized product recommendations for any product based on user preferences""",
            backstory="""You are the best product recommendation agent in the world. 
            You have knowledge about any product the customer asks you for. Your recommendations are logical and can convince
            the people who are looking for the product. Don't ask the user what brand they want to buy. Suggest everything you know using the tools you have.""",
            tool=[SerperDevTool],
            verbose=True,
            llm=llm,
            allow_delegation=True
        )

        st.write("### Enter a Text Prompt")
        text_prompt = st.text_input("Your Text Prompt")
        recommendation_task = Task(
            description=f'Give the best possible product recommendations of {text_prompt} to customers based on their preferences',
            agent=recommendation_agent,
            tool=[SerperDevTool],
            expected_output='output which is expected in string'
        )

        crew = Crew(
            tasks=[recommendation_task],
            agents=[recommendation_agent],
            verbose=True
        )

        if text_prompt:
            st.write("You entered:", text_prompt)
            result = crew.kickoff()
            st.write(result)

    elif model_type == "Web Searching Model":
        load_dotenv()

        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

        search_agent = Agent(
            role="Web Search Agent",
            goal="To search for information about the given query on the web",
            backstory="You are a web search agent with the ability to gather relevant information from the internet to answer queries. Use the provided tools to find the most accurate and comprehensive answers. Suggest relevant websites where the user can buy the product they asked for and provide an approximate price.",
            tool=[SerperDevTool, WebsiteSearchTool],
            verbose=True,
            llm=llm,
            allow_delegation=True
        )

        st.write("### Enter a Search Query")
        search_query = st.text_input("Your Search Query")

        search_task = Task(
            description=f'Search for information related to the query: "{search_query}"',
            agent=search_agent,
            tool=[SerperDevTool, WebsiteSearchTool],
            expected_output="Relevant information found on the web to answer the query"
        )

        crew = Crew(
            tasks=[search_task],
            agents=[search_agent],
            verbose=True
        )

        if search_query:
            st.write("Searching for:", search_query)
            result = crew.kickoff()
            st.write(result)

if __name__ == '__main__':
    main()