# Project Description

TU Delft Aerospace Engineering Thesis Project. The goal of the project is to create a LLM agent that would positivelly contribute to air traffic operators and pilots. It should act as a virtual assistant to the tower, detect conflicts, resolve conflicts and prevent communication errors or any communication confusions


# Code Instructions

llm/groq_agent.py

This script contains the LLM (on Groq) agent that interacts with the BlueSky.

Guide to use the script:
1. activate virtual environment. If you haven't created on yet, you can install necesarry packages with "pip install -r requirements.txt"
3. run bluesky with from a bluesky folder with "python BlueSky.py". This will start the simulation
4. run the groq_agent.py script from llm folder. Optional arguments: --temperature, --model_name, --system_prompt. Example: "python groq_agent.py --temperature 0.3 --model_name llama3-70b-8192 --system_prompt You are a helpful assistant"
5. The agent will start interacting with the BlueSky simulation. You can see the agent's responses and actions in the terminal. Currently full chat history will be preserved in the agent memory, but only Human Messages and AI final Messages, no intermediate steps.
