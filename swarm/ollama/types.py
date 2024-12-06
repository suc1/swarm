from pydantic import BaseModel
from typing import List, Callable, Union

AgentFunction = Callable[[], Union[str, "Agent", dict]]


class AgentOllama(BaseModel):
    name: str = "Agent"
    model: str = "llama3.2:3b"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True
