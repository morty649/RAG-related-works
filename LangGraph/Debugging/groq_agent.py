from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import START,END
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
import os
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

model = init_chat_model("groq:llama-3.3-70b-versatile")

def make_default_graph():
    graph_workflow = StateGraph(State)

    def call_model(state):
        return {'messages':[model.invoke(state['messages'])]}
    
    graph_workflow.add_node("agent",call_model)
    graph_workflow.add_edge(START,"agent")
    graph_workflow.add_edge("agent",END)

    agent = graph_workflow.compile()
    return agent

def make_alt_graph():
    """Making  a tool-calling agent"""
    @tool
    def add(a:float,b:float):
        """Adds two numbers"""
        return a+b
    
    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])
    def call_model(state):
        return {'messages':[model_with_tools.invoke(state['messages'])]}
    
    def should_continue(state:State):
        if state["messages"][-1].tool_calls:
            return "tools"
        return END
    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent",call_model)
    graph_workflow.add_node("tools",tool_node)
    graph_workflow.add_edge("tools","agent")
    graph_workflow.add_edge(START,"agent")
    graph_workflow.add_edge("agent",END)
    graph_workflow.add_conditional_edges("agent",should_continue)

    agent = graph_workflow.compile()
    return agent
    


agent = make_default_graph()

