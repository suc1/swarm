from swarm import Swarm

onLineLLM = False
if onLineLLM:
    from swarm import Agent
    client = Swarm()
else:
    from swarm.ollama.types import AgentOllama as Agent
    from swarm.ollama.clientOllama import GetClientOllama

    clientLLM = GetClientOllama(base_url="http://localhost:11434")
    client = Swarm(client=clientLLM)

def transfer_to_agent_b():
    return agent_b


agent_a = Agent(
    name="Agent A",
    instructions="You are a helpful agent.",
    functions=[transfer_to_agent_b],
)

agent_b = Agent(
    name="Agent B",
    instructions="Only speak in Chinese.",
)

response = client.run(
    agent=agent_a,
    messages=[{"role": "user", "content": "I want to talk to agent B."}],

    model_override="gpt-4o-mini" if onLineLLM else "llama3.2:1b",       # https://ollama.com/blog/tool-support
)

print(response.messages[-1]["content"])
