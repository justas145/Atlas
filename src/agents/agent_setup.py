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
    planner_system_prompt,
    executor_system_prompt,
    verifier_system_prompt,
    single_agent_system_prompt,
    seperation_guidelines,
    bluesky_commands,
    operator_preference,
    experience_lib_instructions,
    planner_options,
    examples,
)
from typing import Optional
from langchain_core.messages import AIMessageChunk
from utils.speak_text import speak_text
from voice_assistant.config import Config



def setup_chat_model(config, groq_api_key):
    if "gpt" in config["model_name"]:
        print('Using OpenAI model')
        chat = ChatOpenAI(
            temperature=config["temperature"], model_name=config["model_name"]
        )

    else:
        print('Using Groq model')
        chat = ChatGroq(
            temperature=config["temperature"],
            model_name=config["model_name"],
            api_key=groq_api_key,
        )
    return chat


def setup_single_agent(config, groq_api_key):
    # Load base system prompt
    system_prompt = hub.pull("hwchase17/openai-tools-agent")

    # Update the system prompt using the single_agent_system_prompt
    system_prompt.messages[0].prompt.template = single_agent_system_prompt.format(
        seperation_guidelines=seperation_guidelines,
        experience_lib_instructions=(
            experience_lib_instructions if config.get("use_skill_lib", False) else ""
        ),
        bluesky_commands=bluesky_commands,
        examples=examples,
        operator_preference=operator_preference,
    )

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
        # Exclude "SEARCHEXPERIENCELIBRARY" tool if 'use_skill_lib' is False
        tools_to_use = [
            tool
            for name, tool in agent_tool_dict.items()
            if name != "SEARCHEXPERIENCELIBRARY"
        ]
        print("not using skill lib")
    # Create and return the agent
    agent = agents.create_openai_tools_agent(chat, tools_to_use, system_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools_to_use, verbose=True, return_intermediate_steps=True
    )
    return agent_executor


def setup_agent(config, groq_key_generator, client, collection):
    initialize_client_for_tools(client)
    initialize_collection_for_tools(collection)
    if "multi" in config["type"]:
        return setup_multi_agent(config, groq_key_generator)
    elif "single" in config["type"]:
        groq_api_key = next(groq_key_generator)
        return setup_single_agent(config, groq_api_key)
    else:
        raise ValueError(f"Invalid agent type: {config['type']}")

def setup_multi_agent(config, groq_key_generator):
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
        system_prompt = hub.pull("hwchase17/openai-tools-agent")

        if role == "planner":
            system_prompt.messages[0].prompt.template = (
                planner_system_prompt.format(
                    planner_options=planner_options,
                    experience_lib_instructions=(
                        experience_lib_instructions
                        if config.get("use_skill_lib", False)
                        else ""
                    ),
                    seperation_guidelines=seperation_guidelines,
                    examples=examples,
                    operator_preference=operator_preference,
                )
            )
        elif role == "executor":
            system_prompt.messages[0].prompt.template = executor_system_prompt.format(
                bluesky_commands=bluesky_commands
            )
        elif role == "verifier":
            system_prompt.messages[0].prompt.template = verifier_system_prompt.format(
                planner_options=planner_options,
                experience_lib_instructions=(
                    experience_lib_instructions
                    if config.get("use_skill_lib", False)
                    else ""
                ),
                examples=examples,
                seperation_guidelines=seperation_guidelines,
                operator_preference=operator_preference,
            )

        groq_api_key = next(groq_key_generator)
        chat = setup_chat_model(config, groq_api_key)
        tools_to_use = [agent_tool_dict[name] for name in tool_names]
        if config.get("use_skill_lib", False) and role in ["planner", "verifier"]:
            print("using skill lib")
            tools_to_use.append(agent_tool_dict["SEARCHEXPERIENCELIBRARY"])
        else:
            print("not using skill lib")

        agent = agents.create_openai_tools_agent(chat, tools_to_use, system_prompt)
        agents_dict[role] = agents.AgentExecutor(
            agent=agent,
            tools=tools_to_use,
            verbose=True,
            return_intermediate_steps=True,
        )

    return MultiAgent(agents_dict, agents_prompts)


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


def process_agent_output(agent_executor, user_input, voice_mode, voice=None):
    result = None
    if isinstance(agent_executor, MultiAgent):
        # Handle MultiAgent case
        result = agent_executor.invoke({"input": user_input}, voice_mode)
        return result["output"] if isinstance(result, dict) and "output" in result else result
    else:
        # Handle single agent case
        if voice_mode in ['1-way', '2-way']:
            for step in agent_executor.iter({"input": user_input}):
                if isinstance(step, dict) and 'intermediate_step' in step:
                    for intermediate_step in step['intermediate_step']:
                        if isinstance(intermediate_step, tuple) and len(intermediate_step) == 2:
                            action, result = intermediate_step
                            for message in action.message_log:
                                if isinstance(message, AIMessageChunk) and message.content:
                                    speak_text(message.content, voice_mode, voice)
                elif isinstance(step, dict) and "output" in step:
                    speak_text(step['output'], voice_mode, voice)
                    result = step['output']
        else:
            result = agent_executor.invoke({"input": user_input})
            if isinstance(result, dict) and "output" in result:
                result = result["output"]
    return result


class MultiAgent:
    def __init__(self, agents, prompts):
        self.agents = agents
        self.prompts = prompts
    def invoke(self, input_dict, voice_mode=None):
        user_input = input_dict.get("input", "")
        planner_prompt = self.prompts["planner"].format(user_input=user_input)
        print("Planner Agent Running")
        planner_voice = Config.MULTI_AGENT_VOICES.get("planner", None)
        plan = process_agent_output(
            self.agents["planner"], planner_prompt, voice_mode, voice=planner_voice
        )

        plan_exists = extraction_plan_runnable.invoke({"text": plan}).plan_exists

        if not plan_exists:
            return {"output": plan, "status": "resolved"}

        while True:
            controller_prompt = self.prompts["executor"].format(plan=plan)
            print("Controller Agent Running")
            controller_voice = Config.MULTI_AGENT_VOICES.get("executor", None)
            controller_output = process_agent_output(self.agents["controller"], controller_prompt, voice_mode, voice=controller_voice)

            verifier_prompt = self.prompts["verifier"].format(
                user_input=user_input, plan=plan
            )
            print("Verifier Agent Running")
            verifier_voice = Config.MULTI_AGENT_VOICES.get("verifier", None)
            verifier_output = process_agent_output(self.agents["verifier"], verifier_prompt, voice_mode, voice=verifier_voice)

            plan_exists = extraction_plan_runnable.invoke(
                {"text": verifier_output}
            ).plan_exists
            if not plan_exists:
                return {"output": verifier_output, "status": "resolved"}
            else:
                plan = verifier_output
