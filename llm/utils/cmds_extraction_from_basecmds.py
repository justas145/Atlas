import os
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# load env
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

output_parser = StrOutputParser()
llm = ChatOpenAI(openai_api_key = openai_api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

chain = prompt | llm | output_parser

# make langchain prompt with formating

prompt_template = PromptTemplate.from_template(
    "Here's the content of a text file describing a command for a system:\n\n{txt}\n\nBased on this description, format a concise summary in the format command - description, where the description is limited to 10 words."
)


def prepare_file_content_for_llm(directory_path):
    prepared_contents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r') as file:
                content = file.read()
                # Prepare the content in a format suitable for an LLM prompt
                input = prompt_template.format(txt=content)
                answer = chain.invoke({"input": input})
                print(answer)
                prepared_contents.append(answer)    
    return prepared_contents


# Example usage
directory_path = '../skills-library/description/'
prepared_contents = prepare_file_content_for_llm(directory_path)
prepared_contents_file_path = '../prompts/basecmds.txt'
# save prepared content to txt file. each new item is a new line
with open('prepared_contents.txt', 'w') as file:
    for content in prepared_contents:
        file.write(content + '\n')

# Example of how you would use the prepared content with an LLM
for content in prepared_contents:
    # You would replace this print statement with the LLM processing step
    print(content)
