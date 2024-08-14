import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from crewai_tools import SerperDevTool, WebsiteSearchTool
from crewai import Agent, Task, Crew
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


st.title("Web Search Agent")

st.write("### Enter a Search Query")
search_query = st.text_input("Your Search Query")

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyA_6sGQ0XzmUaGuWX85gpmn19vrvlmPJhM")

search_agent = Agent(
    role="Web Search Agent",
    goal="To search information about the given query on the web",
    backstory="You are a web search agent with the ability to gather relevant information from the internet to answer queries. Use the provided tools to find the most accurate and comprehensive answers. suggest relevant website where the user can buy the product he/she asked for. Also give an approximate price for it",
    tool=[SerperDevTool, WebsiteSearchTool],
    verbose=True,
    llm=llm,
)

search_task = Task(
    description='Search for information related to the query: "{}"'.format(search_query),
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