import os
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain
from langchain.agents import initialize_agent, Tool
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from googleapiclient.discovery import build  # Google API client
from constants import huggingface_key, google_api_key, cse_id  # Importing the API keys from constants.py

# Set Hugging Face API key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_key

# Initialize Hugging Face pipeline
pipe = pipeline("text2text-generation", model="google/flan-t5-large")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# Wrap Hugging Face pipeline in LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Custom Google search function using the Google Custom Search API
def search_with_google(query: str):
    service = build("customsearch", "v1", developerKey=google_api_key)
    result = service.cse().list(q=query, cx=cse_id, num=5).execute()  # Limiting to 5 results
    search_items = result.get("items", [])
    return "\n".join([f"{i+1}. {item['title']} - {item['link']}" for i, item in enumerate(search_items)])

# Initialize research tools
wikipedia_tool = WikipediaQueryRun()  # Correct import for Wikipedia tool

# Define tools for LangChain agent
tools = [
    Tool(name="Wikipedia", func=wikipedia_tool.run, description="Search for facts on Wikipedia."),
    Tool(name="Search Engine", func=search_with_google, description="Search the web using Google.")
]

# Initialize the agent with the tools
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)

# Define Prompt Templates for blog sections
heading_template = PromptTemplate(
    input_variables=['topic'],
    template="Generate a catchy and informative heading for a blog about {topic}."
)

introduction_template = PromptTemplate(
    input_variables=['topic'],
    template="Write an engaging introduction to the blog about {topic}."
)

content_template = PromptTemplate(
    input_variables=['topic'],
    template="Generate detailed and well-researched content for a blog on {topic}. Include references and sources."
)

summary_template = PromptTemplate(
    input_variables=['topic'],
    template="Summarize the main points covered in a blog about {topic}."
)

# Memory to store conversation history
topic_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

# Create LLM chains for each section of the blog
heading_chain = LLMChain(
    llm=llm,
    prompt=heading_template,
    verbose=True,
    output_key='heading',
    memory=topic_memory
)

introduction_chain = LLMChain(
    llm=llm,
    prompt=introduction_template,
    verbose=True,
    output_key='introduction',
    memory=topic_memory
)

content_chain = LLMChain(
    llm=llm,
    prompt=content_template,
    verbose=True,
    output_key='content',
    memory=topic_memory
)

summary_chain = LLMChain(
    llm=llm,
    prompt=summary_template,
    verbose=True,
    output_key='summary',
    memory=topic_memory
)

# Sequential Chain combining the blog generation steps
blog_chain = SequentialChain(
    chains=[heading_chain, introduction_chain, content_chain, summary_chain],
    input_variables=['topic'],
    output_variables=['heading', 'introduction', 'content', 'summary'],
    verbose=True
)

# Streamlit App
st.title("Blog Generation System")

# Input for the blog topic
input_topic = st.text_input("Enter the blog topic you want to generate:")

# Running the blog generation system
if input_topic:
    # Run the agent to gather research
    research = agent.run(f"Research the topic: {input_topic}")
    
    # Generate the blog sections
    response = blog_chain({'topic': input_topic})
    
    # Displaying the results in Streamlit
    st.subheader("Generated Blog")

    st.write("**Heading:**")
    st.write(response['heading'])

    st.write("**Introduction:**")
    st.write(response['introduction'])

    st.write("**Content:**")
    st.write(response['content'])

    st.write("**Summary:**")
    st.write(response['summary'])

    st.write("**Research Notes:**")
    st.write(research)

    # Expanding sections to show detailed information
    with st.expander("View conversation history"):
        st.info(topic_memory.buffer)
