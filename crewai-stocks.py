import json
import os
from datetime import datetime
import yfinance as yf
from crewai import  Agent, Task, Crew
from crewai.process import Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from dotenv import load_dotenv

import streamlit as st

# In[ ]:


def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock    

yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches stocks prices for {ticket} from the last year about a specific stock",
    func = lambda ticket: fetch_stock_price(ticket)
)



#import llm from openAi
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")


# In[6]:


# first agent
stockPriceAnalyst = Agent(
    role = "Senior Stock price Analyst",
    goal = "Find the {ticket} stock price and analysis trends",
    backstory = """ You're a highly experienced in analysing the price of 
    an especific stock and make predictions about its future price.""" ,
    verbose = True,
    llm = llm,
    max_iter = 15,
    # max_rpm=15,
    # max_execution_time=30,
    memory = True,
    tools = [yahoo_finance_tool],
    allow_delegation = False
)


# In[7]:


getStockPrice = Task(
    description = "Analyze the stock {ticket} price history and create a trend analyses of up, down or sideways",
    agent = stockPriceAnalyst,
    expected_output =  """ Specify the current trend stock price - up, down or sideways, Eg. stock= 'APPL, price UP' """,
)


# In[8]:


# importing the search tool
search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)



# In[9]:


newsAnalyst = Agent(
    role = "Stock News Analyst",
    goal = """Create a short summary of the market news related to stock {ticket} company.
     Specify the current trend - up, down or sideways taking into consideration the news context. 
     For each request stockt asset, specify a number within 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
    backstory = """ 
        You're a highly experienced in analysing market trends and news and have tracked assets for more then 10 years.
        You're also a master level analyst in thre tradicional markets and have deep fundamentalist understanding of the financial information of the companies.
        You understand news, theirs titles and information, but you look at then with a healthy dose of skepticism.
        You consider also the source of the news articles, ignoring the non reputable sources.
     """ ,
    verbose = True,
    llm = llm,
    max_iter = 15,
    max_rpm=15,
    # max_execution_time=30,
    memory = True,
    tools = [search_tool],
    allow_delegation = False
)


# In[10]:


getNews = Task(
    description = f"""
    Take the stock and always include ADA(from Cardano) and BTC to it(if not requested). 
    Use the search tool to search each one individually.

    the current date is {datetime.now()}. 

    Compose the results into a helpfull report.
    
    """,
    agent = newsAnalyst,
    expected_output =  """ A summary of the overall market and one sentence summary for each
    requested asset.
    Include a fear/greed score for each asset based on the news. 
    Use the format: 
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,
)


# In[11]:


stockAnalystReporter = Agent(
    role = "Senior Stock Analyst Writer",
    goal = """
    Write an insightfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend.
        """,
    backstory = """ 
        You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stores
        and narratives that are solid. You follow the investing phylosophy of Howard Marks.
        You are able to hold multiple options when analysing anything.
     """ ,
    verbose = True,
    llm = llm,
    max_iter = 5,
    memory = True,
    allow_delegation = True
)


# In[12]:


writeAnalyses = Task(
    description = """
        Use the stock price trend and the stock news report to create an analyses 
        and write the newsletter about the {ticket} company.
        Be brief and highlight the most important points.
        Focus on the stock price trend, Price-to-Earnings (P/E) ratio , news and fear/greed score.
        What are the near future considerations?
        Include the previous analyses of stock trend and news summary.
    """,
    agent = stockAnalystReporter,
    expected_output = """
        An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner.
        It should contain:
            - bullet points for executive summary
            - Introduction - set the overall picture and spike up the interest
            - main part provides the rest of the analysis including the news summary and fear/greed scores
            - summary - key facts and concrete future trend prediction - up, down and sideways.
            - Price-to-Earnings (P/E) ratio relation to stock price - cheap, regular or expensive.
    """,
    
    context = [getStockPrice, getNews]
)   


# In[13]:


crew = Crew(
    agents = [stockPriceAnalyst, newsAnalyst, stockAnalystReporter],
    tasks = [getStockPrice, getNews, writeAnalyses],
    verbose = True,
    process= Process.hierarchical,
    full_output = True,
    share_crew = False,
    manager_llm = llm,
    max_iter = 15,
    max_rpm=15,
    # max_execution_time=30
)


# In[17]:




# In[19]:


with st.sidebar:
    st.header('Enter the ticket of the stock')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label="Run Research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket")
    else:
        results = crew.kickoff(inputs={'ticket':topic})

        st.subheader("Results of your research:")
        st.write(results['final_output'])

        



