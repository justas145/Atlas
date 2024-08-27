from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain import agents, hub
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentExecutor
from tools.agent_tools import (
    agent_tool_dict,
    initialize_client_for_tools,
    initialize_collection_for_tools,
)
from utils.load_apis_and_configs import cyclic_key_generator
from prompts.agent_prompts import (
    planner_prompt,
    executor_prompt,
    verifier_prompt,
    extraction_prompt,
)
from typing import Optional


def setup_chat_model(config, groq_api_key):
    if "gpt" in config["model_name"]:
        chat = ChatOpenAI(
            temperature=config["temperature"], model_name=config["model_name"]
        )
    else:
        chat = ChatGroq(
            temperature=config["temperature"],
            model_name=config["model_name"],
            api_key=groq_api_key,
        )
    return chat


def setup_single_agent(config, groq_api_key):
    # Load system prompts
    prompt = hub.pull("hwchase17/openai-tools-agent")
    if config.get("use_skill_lib", False):
        print("loading system prompt with exp lib")
        with open("prompts/system_with_exp_lib.txt", "r") as f:
            prompt.messages[0].prompt.template = f.read()
    else:
        with open("prompts/system.txt", "r") as f:
            prompt.messages[0].prompt.template = f.read()
    print("setting up agent with config:", config)
    # Initialize LLM model
    chat = setup_chat_model(config, groq_api_key)
    # Decide on tools to include based on the 'use_skill_lib' configuration
    if config.get("use_skill_lib", False):
        tools_to_use = list(
            agent_tool_dict.values()
        )  # Use all available tools if skill lib is to be used
        print("using skill lib")
    else:
        # Exclude "QueryConflicts" tool if 'use_skill_lib' is False
        tools_to_use = [
            tool
            for name, tool in agent_tool_dict.items()
            if name != "SEARCHEXPERIENCELIBRARY"
        ]
        print("not using skill lib")
    # Create and return the agent
    agent = agents.create_openai_tools_agent(chat, tools_to_use, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools_to_use, verbose=True, return_intermediate_steps=True
    )
    return agent_executor


def setup_agent(config, groq_api_key_lst, client, collection):
    initialize_client_for_tools(client)
    initialize_collection_for_tools(collection)
    groq_key_generator = cyclic_key_generator(groq_api_key_lst)
    if "multi" in config["type"]:
        return setup_multi_agent(config, groq_api_key_lst)
    elif "single" in config["type"]:
        groq_api_key = next(groq_key_generator)
        return setup_single_agent(config, groq_api_key)
    else:
        raise ValueError(f"Invalid agent type: {config['type']}")


def setup_multi_agent(config, groq_api_keys_lst):
    print("setting up multi agent with config:", config)
    agents_dict = {}
    tool_sets = {
        "planner": ["GETALLAIRCRAFTINFO", "CONTINUEMONITORING"],
        "controller": ["SENDCOMMAND"],
        "verifier": ["GETALLAIRCRAFTINFO", "CONTINUEMONITORING"],
    }

    agents_prompts = {
        "planner": planner_prompt,
        "executor": executor_prompt,
        "verifier": verifier_prompt,
    }

    for role, tool_names in tool_sets.items():
        prompt_init = hub.pull("hwchase17/openai-tools-agent")
        ### LOADING PROMPTS ###
        if role == "planner":
            if config.get("use_skill_lib", False):
                print("loading system prompt with exp lib")
                with open("prompts/planner_system_with_exp_lib.txt", "r") as f:
                    planner_system_prompt = prompt_init
                    planner_system_prompt.messages[0].prompt.template = f.read()
            else:
                with open("prompts/planner_system.txt", "r") as f:
                    planner_system_prompt = prompt_init
                    planner_system_prompt.messages[0].prompt.template = f.read()
        if role == "executor":
            with open("prompts/executor_system.txt", "r") as f:
                executor_system_prompt = prompt_init
                executor_system_prompt.messages[0].prompt.template = f.read()
        if role == "verifier":
            if config.get("use_skill_lib", False):
                print("loading system prompt with exp lib")
                with open("prompts/verifier_system_with_exp_lib.txt", "r") as f:
                    verifier_system_prompt = prompt_init
                    verifier_system_prompt.messages[0].prompt.template = f.read()
            else:
                with open("prompts/verifier_system.txt", "r") as f:
                    verifier_system_prompt = prompt_init
                    verifier_system_prompt.messages[0].prompt.template = f.read()
        ### LOADING PROMPTS ###
        groq_key_generator = cyclic_key_generator(groq_api_keys_lst)
        groq_api_key = next(groq_key_generator)
        chat = setup_chat_model(config, groq_api_key)
        tools_to_use = [agent_tool_dict[name] for name in tool_names]
        if config.get("use_skill_lib", False) and role in ["planner", "verifier"]:
            print("using skill lib")
            tools_to_use.append(agent_tool_dict["SEARCHEXPERIENCELIBRARY"])
        print("not using skill lib")

        if role == "planner":
            prompt = planner_system_prompt
        elif role == "executor":
            prompt = executor_system_prompt
        elif role == "verifier":
            prompt = verifier_system_prompt
        agent = agents.create_openai_tools_agent(chat, tools_to_use, prompt)
        agents_dict[role] = agents.AgentExecutor(
            agent=agent,
            tools=tools_to_use,
            verbose=True,
            return_intermediate_steps=True,
        )

    with open("prompts/ICAO_seperation_guidelines.txt", "r") as f:
        icao_seperation_guidelines = f.read()

    return MultiAgent(agents_dict, agents_prompts, icao_seperation_guidelines)


class PlanExists(BaseModel):
    """Information about an aircraft conflict resolution plan."""

    # This doc-string is sent to the LLM as the description of the schema Metadata,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    plan_exists: Optional[bool] = Field(
        default=None,
        description="Whether an aircraft conflict resolution plan exists. Plan typically includes aircraft call signs and instructions for that aircraft. If not provided, set to False. If provided, set to True. If the plan is provided and there is a comment that there are no conflicts or any other comments, set to True.",
    )


extraction_plan_llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini")
extraction_plan_runnable = (
    extraction_prompt | extraction_plan_llm.with_structured_output(schema=PlanExists)
)


class MultiAgent:

    def __init__(self, agents, prompts, icao_seperation_guidelines):
        self.agents = agents
        self.prompts = prompts
        self.icao_seperation_guidelines = icao_seperation_guidelines

    def invoke(self, initial_input):
        planner_prompt = self.prompts["planner"].format(
            icao_seperation_guidelines=self.icao_seperation_guidelines
        )
        print("Planner Agent Running")
        plan = self.agents["planner"].invoke({"input": planner_prompt})["output"]

        plan_exists = extraction_plan_runnable.invoke({"text": plan}).plan_exists

        if not plan_exists:
            return {"output": plan, "status": "resolved"}

        while True:
            controller_prompt = self.prompts["executor"].format(plan=plan)
            print("Controller Agent Running")
            controller_output = self.agents["controller"].invoke(
                {"input": controller_prompt}
            )["output"]

            verifier_prompt = self.prompts["verifier"].format(
                icao_seperation_guidelines=self.icao_seperation_guidelines, plan=plan
            )
            print("Verifier Agent Running")
            verifier_output = self.agents["verifier"].invoke(
                {"input": verifier_prompt}
            )["output"]

            plan_exists = extraction_plan_runnable.invoke(
                {"text": verifier_output}
            ).plan_exists
            if not plan_exists:
                return {"output": verifier_output, "status": "resolved"}
            else:
                plan = verifier_output
