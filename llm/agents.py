# agents.py
from langchain import agents, hub


class BaseAgent:
    def __init__(self, temperature, model_name, prompt_path):
        self.temperature = temperature
        self.model_name = model_name
        self.prompt_path = prompt_path
        self.agent_executor = None
        self.initialize_agent()

    def initialize_agent(self):
        prompt = hub.pull("hwchase17/openai-tools-agent")
        with open(self.prompt_path, "r") as f:
            prompt.messages[0].prompt.template = f.read()

        chat = ChatGroq(temperature=self.temperature, model_name=self.model_name)
        agent = agents.create_openai_tools_agent(chat, agent_tools_list, prompt)
        self.agent_executor = agents.AgentExecutor(
            agent=agent, tools=agent_tools_list, verbose=True
        )

    def execute(self, input_data):
        return self.agent_executor.invoke({"input": input_data})


class SingleAgent(BaseAgent):
    pass  # Additional behaviors or overrides specific to single agents


class MultiAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sub_agents = []  # List to manage multiple agents if necessary

    def add_agent(self, agent):
        self.sub_agents.append(agent)

    def execute(self, input_data):
        # Example of a multi-agent execution flow
        output_data = input_data
        for agent in self.sub_agents:
            output_data = agent.execute(output_data)
        return output_data


# Example usage
if __name__ == "__main__":
    single_agent = SingleAgent(0.3, "llama3-70b-8192", "prompts/system.txt")
    multi_agent = MultiAgent(0.3, "llama3-70b-8192", "prompts/system.txt")
    multi_agent.add_agent(SingleAgent(0.3, "llama3-70b-8192", "prompts/conflict.txt"))

    # Example execution
    print(single_agent.execute("solve aircraft conflict."))
    print(multi_agent.execute("solve aircraft conflict."))
