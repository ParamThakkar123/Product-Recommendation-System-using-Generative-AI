import streamlit as st
from PIL import Image
import PyPDF2
import os
from crewai_tools import SerperDevTool, WebsiteSearchTool, ScrapeWebsiteTool
from crewai import Agent, Task, Crew
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

def main():
    st.title("Hyperpersonalization and Prompt based shopping experience using generative AI")

    st.sidebar.header("Upload files")

    input_type = st.sidebar.selectbox(
        "Choose the type of file you want to upload",
        ("Image", "PDF")
    )

    # Conditional file upload based on selected input type
    if input_type == "Image":
        image_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if image_file is not None:
            image = Image.open(image_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

    elif input_type == "PDF":
        pdf_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
        if pdf_file is not None:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            st.write(f"Number of pages in the uploaded PDF: {num_pages}")
            for page in pdf_reader.pages:
                st.text(page.extract_text())

    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    # llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"])
    # image_llm = ('gemini-1.5-flash')

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro", verbose=True, temperature=0.9, google_api_key=os.environ['GEMINI_API_KEY']
    )

    recommendation_agent = Agent(
        role="Product Recommendation Agent",
        goal="""To give personalized product recommendations for any product based on user preferences""",
        backstory="""You are the best product recommendation agent in the world. 
        You have knowledge about any product the customers asks you a recommendation for.
        Your recommendations are logical and can convince the people who are looking for the 
        product. Don't ask user what brand they want to buy. Suggest everything you know using the tools you have.
        you can also search over the web to give the user sites where they can buy the items you suggested.
        Also show the price. You can scrape a website to get the price data.
        """,
        tool=[SerperDevTool, WebsiteSearchTool, ScrapeWebsiteTool],
        verbose=True,
        llm=llm,
    )

    # image_qa_agent = Agent(
    #     role="Image Question Answering Agent",
    #     goal="""To answer questions based on the image.""",
    #     backstory="""You are a Question Answering agent who will answer questions of users based on the image given.
    #     The image can be of a something and if the user asks you about product recommendations from that image, you can give product recommendations
    #     of similar or better products present in the image. You always give the best possible response to all user queries""",
    #     tool=[SerperDevTool, WebsiteSearchTool, ScrapeWebsiteTool],
    #     verbose=True,
    #     llm=image_llm
    # )

    web_search_agent = ""

    # Text input at the bottom

    st.write("### Enter a Text Prompt")
    text_prompt = st.text_input("Your Text Prompt")
    recommendation_task = Task(
        description='Give the best possible product recommendations of {product} to customers based on their preferences'.format(product=text_prompt),
        agent=recommendation_agent,
        tool=[SerperDevTool, WebsiteSearchTool, ScrapeWebsiteTool]
    )

    # image_qa_task = Task(
    #     description='To answer all the questions asked by the user in the best way possible',
    #     agent=image_qa_agent,
    #     tool=[SerperDevTool, WebsiteSearchTool, ScrapeWebsiteTool]
    # )

    crew = Crew(
        tasks=[recommendation_task],
        agents=[recommendation_agent],
        verbose=True
    )

    if text_prompt:
        st.write("You entered:", text_prompt)
        result = crew.kickoff()
        st.write(result)

    if text_prompt and input_type == "Image":
        st.write("You entered:", text_prompt)
        result = crew.kickoff()
        st.write(result)
        

if __name__ == '__main__':
    main()