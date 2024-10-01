import streamlit as st
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from crewai_tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import numpy as np
import requests
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

# Precision@K
def precision_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    intersection = actual_set.intersection(predicted_set)
    return len(intersection) / min(k, len(predicted_set))

# Recall@K
def recall_at_k(actual, predicted, k):
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    intersection = actual_set.intersection(predicted_set)
    return len(intersection) / len(actual_set)

# NDCG@K
def dcg_at_k(relevant, k):
    relevant = np.array(relevant)[:k]
    return np.sum((2**relevant - 1) / np.log2(np.arange(2, relevant.size + 2)))

def ndcg_at_k(actual, predicted, k):
    actual_relevance = [1 if item in actual else 0 for item in predicted[:k]]
    ideal_relevance = sorted(actual_relevance, reverse=True)
    dcg = dcg_at_k(actual_relevance, k)
    idcg = dcg_at_k(ideal_relevance, k)
    return dcg / idcg if idcg > 0 else 0

@tool
def calculate_metrics(actual, predicted, k):
    """
    This CrewAI tool calculates Precision@K, Recall@K, and NDCG@K.

    Args:
        actual: A list of actual relevant items.
        predicted: A list of predicted recommended items.
        k: The cutoff rank for calculation.

    Returns:
        A string summarizing the results of Precision@K, Recall@K, and NDCG@K.
    """
    precision = precision_at_k(actual, predicted, k)
    recall = recall_at_k(actual, predicted, k)
    ndcg = ndcg_at_k(actual, predicted, k)

    return (
        f"Metrics for Top {k} recommendations:\n"
        f"Precision@{k}: {precision:.4f}\n"
        f"Recall@{k}: {recall:.4f}\n"
        f"NDCG@{k}: {ndcg:.4f}"
    )

os.environ["SERPER_API_KEY"] = "e5d1106caed5747b9774f708088544929d35117a"

st.title("Personalised Multimodal, Multi Agent, Autonomous Recommendation System")
st.sidebar.header("Upload files")

text_prompt = st.text_input("What do you wish to buy today ??")

chat_with_history = MongoDBChatMessageHistory(
    session_id=text_prompt,
    connection_string="mongodb+srv://paramthakkar864:llmresearch@cluster0.lj5ad.mongodb.net/history?retryWrites=true&w=majority&appName=Cluster0",
    database_name="history",
    collection_name="chat_history",
)

if text_prompt:
    chat_with_history.add_user_message(text_prompt)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpeg", "jpg"])
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key="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")
# response = ""
# if uploaded_file is not None:
#     st.write("Multi modal agent started")
#     gemini_api_key = "AIzaSyCyWVCFhbMpfjbZqVnpWEIeTJwBKR1JHZY"
#     genai.configure(api_key=gemini_api_key)
#     model = genai.GenerativeModel("gemini-1.5-pro")
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded image", use_column_width=True)

#     prompt = "Analyse the image and list the products present in the image"
#     response = model.generate_content([image, prompt])


serper_tool = SerperDevTool()

# Product Review Agent
recommendation_agent = Agent(
    role="Product Recommendation Agent",
    goal=f"""To give personalized product recommendations for any product {text_prompt} based on user preferences""",
    backstory=f"""You are the best product recommendation agent in the world.
    You have knowledge about any product the customer asks you for as a {text_prompt}. Your recommendations are logical and can convince
    the people who are looking for the product. Suggest everything you know using the tools you have. Give a list of 10 items for recommendations as per {text_prompt} always. Try to give the best possible answer in no more than 2 queries. Stop as soon as you got the output.""",
    tools=[serper_tool],
    verbose=True,
    memory=True,
    llm=llm,
    allow_delegation=True
)

recommendation_task = Task(
    description=f'Give the best possible product recommendations of {text_prompt} to customers based on their preferences. Fetch data from the web using tools and give the latest information. Give a list of 10 items for recommendations always. Try to give the best possible answer in no more than 2 queries. Stop as soon as you got the output.',
    agent=recommendation_agent,
    tools=[serper_tool],  # Properly initialized tools
    expected_output='output which is expected in string'
)

# Web search agent
# search_agent = Agent(
#     role="Web Search Agent",
#     goal="To search for information about the given query on the web",
#     backstory=f"""You are an expert web search agent with the ability to gather the most relevant and comprehensive information from the internet to answer queries.
#     Be as accurate as possible at suggesting sites. Use the provided tools to find the most accurate answers and suggest reputable websites where the user can purchase the requested product. Give a list of 10 items for recommendations always.
#     Try to give the best possible answer as fast as possible and terminate on completion. Stop as soon as you got the output.""",
#     tools=[serper_tool],  # Properly initialized tools
#     verbose=True,
#     memory=True,
#     llm=llm,
#     allow_delegation=True
# )

# search_task = Task(
#     description=f'Search for information related to the query: "{text_prompt}"',
#     agent=search_agent,
#     tools=[serper_tool],  # Properly initialized tools
#     expected_output="Relevant information found on the web to answer the query"
# )

# System Evaluator agent
# evaluator_agent = Agent(
#     role="LLM Evaluator Agent",
#     goal="To evaluate the recommendations given by all the agents",
#     backstory=f"""You are the best evaluator of AI agents on earth.
#     Your task is to evaluate your fellow AI agents regarding how better they are performing at their task of providing personalised recommendations to users. your task is to take the recommendations list from your fellow AI agents and check if the recommendations are really relevant to the query that is asked. Calculate various metrics for this like Precision@K, Recall@K, Normalized Discounted Cumulative Gain. And give the scores to their prediction. Be as accurate as possible in your calculations.
#     Try to give the best possible answer as fast as possible and terminate on completion.
#     You can use the calculate_metrics tool given to you to make all the calculations.""",
#     verbose=True,
#     memory=True,
#     llm=llm,
#     tools=[serper_tool, calculate_metrics],
#     allow_delegation=True
# )

# evaluation_task = Task(
#     description="Analyse and calculate the metric score for recommendations provided by all the agents.Try to give the best possible answer as fast as possible and terminate on completion.",
#     agent=evaluator_agent,
#     tools=[serper_tool, calculate_metrics],
#     expected_output="A accurately and logically calculated metric scores for the recommendations given by various AI agents."
# )

crew = Crew(
    tasks=[recommendation_task],
    agents=[recommendation_agent],
    planning=True,
    planning_llm=llm
)

if text_prompt:
    result = crew.kickoff()
    st.write(result)
    print(result)
    chat_with_history.add_ai_message(result)

# if uploaded_file is not None:
#     image_analysis_agent = Agent(
#         role="Analysis and Recommendation Expert",
#         goal=f"""Analyze an {response} to answer questions and provide best possible recommendations recommendations.
#                 Try to give the best possible answer as fast as possible and terminate on completion.
#                 Stop as soon as you got the output.""",
#         tools=[serper_tool],  # Properly initialized tools
#         verbose=True,
#         memory=True,
#         backstory="""You are an expert at analyzing and extracting relevant information.
#                     Your task is to help users understand the content and suggest the best products available online.
#                     Give a list of 10 items for recommendations always.""",
#         llm=llm,
#         allow_delegation=True
#     )

#     analyze_image_task = Task(
#         description=f"Based on the image analysis, please address any user questions ({text_prompt}) and provide tailored recommendations for products similar to those inquired about. Suggest web links as accurately as possible. Utilize reputable websites to gather detailed information about these products. Stop as soon as you got the output. Include direct links to these websites for purchasing. Try to give the best possible answer in no more than 2 queries",
#         expected_output="Detailed answers to user questions and a list of recommendations for products with links to where they can be purchased.",
#         agent=image_analysis_agent,
#         tools=[serper_tool]
#     )

#     crew = Crew(
#         tasks=[analyze_image_task, recommendation_task, search_task, evaluation_task],
#         agents=[image_analysis_agent, recommendation_agent, search_agent, evaluator_agent],
#         planning=True,
#         planning_llm=llm
#     )
#     if text_prompt:
#         result = crew.kickoff()
#         st.write(result)
#         print(result)
#         chat_with_history.add_ai_response(result)
