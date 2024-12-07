from pydantic import BaseModel
from typing import List, Callable, Union
from swarm import Agent

AgentFunction = Callable[[], Union[str, "Agent", dict]]


class AgentOllama(Agent):
    model: str = "llama3.2:1b"
