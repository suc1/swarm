from swarm import Swarm, Agent

onLineLLM = False
if onLineLLM:
    modelName = 'gpt-4o-mini'
    client = Swarm()
else:
    modelName = 'llama3.2:1b'       # https://ollama.com/search?c=tools
    from openai import OpenAI
    clientLLM = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
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

    model_override=modelName,
)

print(response.messages[-1]["content"])
