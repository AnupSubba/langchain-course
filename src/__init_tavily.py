from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="URL of the source")

class AgentResponse(BaseModel):
    """Response schema for the agent"""
    answer: str = Field(description="AI response")
    sources: List[Source] = Field(description="List of sources used by the agent")  
    

llm = ChatOpenAI(model="gpt-4o")
tools = [TavilySearch()]

"""agent = create_agent(
    model = llm,
    tools = tools,
    response_format=AgentResponse)"""

agent = create_agent(
    model = llm,
    tools = tools)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Delhi?")]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()