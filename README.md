# Automatic Control With Human-Like Reasoning: Exploring Language Model Embodied Air Traffic Agents

<div align="center">

[![Arxiv](https://img.shields.io/badge/arXiv-2409.09717-b31b1b.svg)](https://arxiv.org/abs/2409.09717)
[![Python Version](https://img.shields.io/badge/Python-≥3.9-blue.svg)](https://github.com/justas145/LLM-Enhanced-ATM)

______________________________________________________________________

[![Watch the video](https://raw.githubusercontent.com/justas145/LLM-Enhanced-ATM/refactor-clean-code/src/results/examples/demo_video_thesis_repository_thumbnail.png)](https://raw.githubusercontent.com/justas145/LLM-Enhanced-ATM/refactor-clean-code/src/results/examples/demo_video_thesis_repository.mp4)

</div>

This project explores the application of a language model-based agent with function-calling and learning capabilities to resolve air traffic conflicts without human intervention. The main components are foundational large language models, tools for simulator interaction, and an innovative experience library. Our study reveals significant performance differences across various configurations of language model-based agents, with the best-performing configuration solving almost all 120 but one imminent conflict scenarios, including up to four aircraft simultaneously.

## Installation

This project requires Python ≥ 3.9

### Setting up the environment

#### For Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### For Mac/Linux (using Conda):

```bash
conda create -n your_env_name python=3.9(or later)
conda activate your_env_name
conda install -c conda-forge portaudio
pip install -r requirements.txt
```

### Setting up .env file

Create a `.env` file in the root directory. Check `template.env` for the required variables.

- You can use [LangSmith](https://www.smith.langchain.com/) for tracking every step of an agent. Get a free API key and set it as `LANGCHAIN_API_KEY`.
- Set `OPENAI_API_KEY` if you plan to use OpenAI models.
- Set `GROQ_API_KEY` if you want to use open-source models hosted on Groq. It's recommended to generate several API keys with multiple accounts due to strict rate limits for running many scenarios.
- Set `DEEPGRAM_API_KEY` if you want to use Deepgram voice models. You can get a free API key with $200 in free credits.

## Running the Project

1. Start the BlueSky simulator:

   ```bash
   cd bluesky
   python BlueSky.py
   ```

   For headless mode: `python BlueSky.py --headless` (note that this will run the simulation in the background and you will not be able to see the simulation GUI)

2. Open a new terminal window and navigate to the `src` directory:

   ```bash
   cd src
   ```

3. Run the main script:

   ```bash
   python conflict_main.py [options]
   ```
   
### Options for conflict_main.py

- `--model_name`: Specify the official model name you want to use from OpenAI or Groq (e.g., "gpt-4o-2024-08-06", "llama3-70b-8192").
- `--agent_type`: Choose "single_agent" or "multi_agent". For multi-agent, all agents will have the same LLM model (e.g., all GPT-4o or all Llama3).
- `--temperature`: Select the model's temperature. For a multi-agent system, all agents will have the same temperature.
- `--use_skill_lib`: Choose "yes" or "no". If "yes", agents will receive an extra tool to search the vector database and extract experience documents.
- `--scenario`: Load a single scenario or a folder with scenarios. Folders can have nested folders with scenarios. Scenario files are stored in `bluesky/scenario`. If `test.scn` is in `bluesky/scenario`, specify "test.scn". If a folder `TEST/Big` is in `bluesky/scenario`, specify "TEST/Big".
- `--output_csv`: Specify a path relative to `conflict_main.py` to save the results. If not specified, results are not saved to a CSV file. If an existing CSV file is specified, results will be appended.
- `--voice`: Choose "2-way", "1-way", or "no". "no" means no voice interaction. "1-way" means only the AI will talk. "2-way" means you will talk, your speech will be recorded and transcribed, and the AI will respond.
- `--preference`: For ATCO preference testing, select "HDG", "ALT", or "tLOS". You still need to provide user input to instruct agents to only use heading or altitude changes. For "tLOS", instruct the agent to start solving conflicts only when the tLOS is less than the specified `--tlos_threshold` value.
- `--tlos_threshold`: Specify the threshold value when using "tLOS" preference. Not required if "tLOS" is not selected.

You can run multiple agent configurations by providing comma-separated values. For example:

```bash
python conflict_main.py --model_name gpt-4o-2024-08-06,llama3-70b-8192 --temperature 0.9
```

This will run the GPT-4o model with a temperature of 0.9 on a conflict scenario, and then run the Llama3 model with the same temperature on the same conflict scenario (scenario is loaded again from the start).

Another example:

```bash
python conflict_main.py --model_name gpt-4o-2024-08-06,llama3-70b-8192 --agent_type single_agent,multi_agent --use_skill_lib yes
```

This will run a single GPT-4o agent with the skill library, and then run multiple Llama3 agents with the skill library.


All arguments are optional. Default values are:

- `--model_name`: "gpt-4o"
- `--agent_type`: "single_agent"
- `--temperature`: 0.3
- `--use_skill_lib`: "no"
- `--scenario`: "TEST/Big"
- `--output_csv`: None (no saving to CSV)
- `--voice`: "no"
- `--preference`: None
- `--tlos_threshold`: None

Default values are provided if options are not specified.

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{andriuškevičius2024automaticcontrolhumanlikereasoning,
      title={Automatic Control With Human-Like Reasoning: Exploring Language Model Embodied Air Traffic Agents}, 
      author={Justas Andriuškevičius and Junzi Sun},
      year={2024},
      eprint={2409.09717},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2409.09717}, 
}
```
