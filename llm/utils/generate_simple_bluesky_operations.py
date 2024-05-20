import os
import random
import glob
import csv
import argparse
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Define relative paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
description_directory = os.path.join(
    base_dir, 'skills-library', 'description3')
csv_file_path = os.path.join(
    base_dir, 'data', 'bluesky_operations', 'test_bluesky_operations.csv')

# Function to read n random text files and combine them into one string


def combine_random_txt_files(n, directory):
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    selected_files = random.sample(txt_files, n)
    combined_content = ""
    for file in selected_files:
        with open(file, 'r') as f:
            combined_content += f.read() + "\n\n#####"
    return combined_content.strip()


# Simulating the llm.invoke() function
generate_user_input_prompt = PromptTemplate.from_template(
    "You are given a list of documents that describe various commands related to aircraft operations. Based on these documents, create a user input string that uses these commands in a coherent and logical sequence. The user input should reflect a realistic scenario involving aircraft operations. Don't give any hints what commands to use in your input, meaning you cannot use commands words in your generated input. For example if you get such document: 'RESOOFF: Resooff Switch for conflict resolution module. The switchthat will turn OFF the conflict resolution module for particular aircraft that will not avoid others. Usage: RESOOFF[acid]' then you would say something like if there are any aircraft turn off conflict resolution for one aircraft. Or for example if you would get MOVE command doc 'MOVE: Move Instantaneously move an aircraft to a new position. If no values for the altitude, heading, speed and climb rate are provided, the aircraft will keep the old values.Usage: MOVE acid,lat,lon,[alt,hdg,spd,vspd]' then user input would be something like: move aircraft A1 to London (you can use lat lon or a location name).  For aircraft call signs only use these available call signs: A1, A2, A3. There can be multiple documents so you user input would be: user input referring to doc1. Then user input referring doc2. Then ... \n\n Here Are a List of documents: {docs}\n\n Only respond with User Input nothing more. Go!"
)

llm_generate_commands_gpt4o = ChatOpenAI(model='gpt-4o-2024-05-13')
llm_generate_commands_gpt35_turbo = ChatOpenAI()
llm_generate_commands_llama3_70b = ChatGroq(
    temperature=0.1, model_name="llama3-70b-8192")

generate_commands_llm_chain = generate_user_input_prompt | llm_generate_commands_gpt4o | StrOutputParser()

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Generate and process random commands.')
parser.add_argument('--num_generated_commands', type=int,
                    default=50, help='Number of commands to generate')
parser.add_argument('--n_min', type=int, default=1,
                    help='Minimum number of random text files to combine')
parser.add_argument('--n_max', type=int, default=5,
                    help='Maximum number of random text files to combine')

args = parser.parse_args()

# Calculate the number of examples per value of n
n_values = list(range(args.n_min, args.n_max + 1))
examples_per_n = args.num_generated_commands // len(n_values)

# Main loop
for n in n_values:
    for _ in range(examples_per_n):
        docs = combine_random_txt_files(n, description_directory)
        user_input = generate_commands_llm_chain.invoke(docs)

        # Check if the CSV file exists, if not create it with headers
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['user_input', 'docs', 'n_commands']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            # Write the new row
            writer.writerow({'user_input': user_input,
                            'docs': docs, 'n_commands': n})

# Handle any remaining examples if num_generated_commands is not evenly divisible by the range size
remaining_examples = args.num_generated_commands % len(n_values)
if remaining_examples > 0:
    for _ in range(remaining_examples):
        n = random.choice(n_values)
        docs = combine_random_txt_files(n, description_directory)
        user_input = generate_commands_llm_chain.invoke(docs)

        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'user_input': user_input,
                            'docs': docs, 'n_commands': n})
