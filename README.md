# Product Recommendation System using Generative AI

## Description and Features

🔍 **Overview**  
We've developed a web application using **Streamlit**, a Python-based frontend UI framework, incorporating advanced AI technologies like **CrewAI** for multi-agent systems and **Langchain** to leverage the capabilities of **Gemini-1.5-pro** and **LLaMA-70B**.

🤖 **AI Integration**  
The system integrates Large Language Models, LLaMA-70B and Gemini-1.5-pro, functioning as multi-agent AI systems that autonomously retrieve and analyze information from the internet to provide optimal product recommendations. These models do not require specific dataset training, as they fetch necessary information dynamically using various tools.

⚡ **Performance**  
The application uses **Groq API** for LPU inference, significantly reducing model response time from minutes to milliseconds.

👥 **Agents**  
It features three simultaneously operating agents:
- **Product Recommendation Agent**: Suggests the best brands and varieties.
- **Image Question-Answering Model**: Analyzes images and answers queries related to objects within them.
- **Web Search Agent**: Scours the web for products, offering relevant links and approximate prices tailored to the user's country and currency.

## Tech Stack

- **Streamlit** 🖥️
- **Langchain** 🔗
- **CrewAI** 🤝
- **Groq API** 🚀

## Future Scope

🔧 **User Profiles and Preferences**  
Incorporate user profiles that capture individual preferences, purchase history, and browsing behavior. This could refine recommendations and responses, making them more relevant to each user.

🔄 **Adaptive Learning**  
Implement mechanisms where the system learns from user interactions to improve recommendations over time. This could involve reinforcement learning or other adaptive techniques.

## Applications

### E-Commerce and Retail 🛒

- **Product Recommendations**: Offer personalized product recommendations based on preferences, browsing history, and current trends.
- **Price Comparison**: Provide comparative pricing and product availability across different e-commerce platforms.
- **Visual Search**: Allow users to upload images to find similar products or detailed information about items.

### Customer Support 🤖

- **Virtual Shopping Assistant**: Act as a virtual assistant that answers questions, suggests products, and helps users navigate online stores.
- **Automated Query Handling**: Manage customer inquiries and support requests by analyzing and responding to questions in real-time.

### Localized Recommendations 🌍

- **Country and Currency Adaptation**: Offer product recommendations and pricing tailored to the user’s country and currency.
- **Regional Trends**: Incorporate regional trends and preferences to tailor recommendations to local markets.

## Project Setup
```python
# 1. Create the following api keys and include them in .env file
GEMINI_API_KEY='YOUR API KEY HERE'
GROQ_API_KEY='YOUR API KEY HERE'
```

```python
# 2. To install the required dependencies
pip install -r requirements.txt
```
```python
# 3. To run the python file using streamlit
streamlit run main.py
```

## Team Members

- [Param Thakkar](https://github.com/ParamThakkar123) 👤
- [Anushka Yadav](https://github.com/2412anushka) 👤

## Screenshots 📸
![Screenshot 2024-08-15 010319](https://github.com/user-attachments/assets/b994cb9a-a09b-478c-8f8b-5840ed7554fe)
![Screenshot 2024-08-15 010405](https://github.com/user-attachments/assets/b9952192-923e-4972-a2b4-fbdac17991d1)
![Screenshot 2024-08-15 010424](https://github.com/user-attachments/assets/c7603f4b-0e2a-47a6-af47-cce3bab23b76)
![Screenshot 2024-08-15 010501](https://github.com/user-attachments/assets/4b2b55a4-ae5a-470e-a44a-39b7cd82197c)
![Screenshot 2024-08-15 010654](https://github.com/user-attachments/assets/5d94d40e-4e43-4fe3-abe3-f9667ff4b4e9)
![Screenshot 2024-08-15 010757](https://github.com/user-attachments/assets/69020b87-71e2-489b-8793-752c9b882955)

## Demo Video Link 🎥

- [Demo Video](https://youtu.be/Yuke-ewHlcg)
