from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

    
llm = ChatOpenAI(model="gpt-4o")
tools = [TavilySearch()]
agent = create_agent(model = llm, tools = tools)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Delhi?")]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()