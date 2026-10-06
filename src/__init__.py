from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query:str) -> str:
    """Use this tool to search for weather information."""
    print("Searching for ", {query})
    #return "Delhi weather is sunny."
    return tavily.search(query=query)
    
llm = ChatOpenAI(model="gpt-4o")
tools = [search]
agent = create_agent(model = llm, tools = tools)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()