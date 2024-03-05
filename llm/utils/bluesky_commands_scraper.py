import requests
from bs4 import BeautifulSoup
import os

# Base URL for command pages
base_command_url = "https://github.com/TUDelft-CNS-ATM/bluesky/wiki/"
save_dir = "C:\\Users\\justa\\OneDrive\\Desktop\\Developer\\LLM-Enhanced-ATM\\llm\\skills-library\\description"


# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)


def get_soup(url):
    """Fetches the webpage and returns a BeautifulSoup object."""
    response = requests.get(url)
    if response.status_code == 200:
        return BeautifulSoup(response.text, 'html.parser')
    else:
        print(f"Failed to retrieve the webpage: {url}")
        return None


def format_table(table):
    """Formats the table content to maintain structure in plain text."""
    headers = [header.text.strip() for header in table.find_all('th')]
    rows = table.find_all('tr')

    # Calculate column widths
    col_widths = [len(header) for header in headers]
    for row in rows[1:]:  # Skip header row
        for i, cell in enumerate(row.find_all('td')):
            col_widths[i] = max(col_widths[i], len(cell.text.strip()))

    # Create formatted table string
    formatted_table = ''
    header_row = ' | '.join(header.ljust(
        col_widths[i]) for i, header in enumerate(headers))
    separator_row = '-+-'.join('-' * width for width in col_widths)
    formatted_table += f"{header_row}\n{separator_row}\n"

    for row in rows[1:]:
        row_data = [cell.text.strip().ljust(col_widths[i])
                    for i, cell in enumerate(row.find_all('td'))]
        formatted_table += ' | '.join(row_data) + '\n'

    return formatted_table


def extract_content(div):
    """Extracts and formats content from the div, excluding specific links, and including paragraphs, tables, and code blocks."""
    content = ""
    for child in div.descendants:  # Use descendants to deeply iterate through all nested tags
        # Handling paragraphs and headings
        if child.name == "p" or (child.name == "div" and 'markdown-heading' in child.get('class', [])):
            # Check and exclude specific link text
            if child.find('a', {'class': 'internal present', 'href': '/TUDelft-CNS-ATM/bluesky/wiki/Command-Reference'}):
                continue  # Skip adding this element's content
            content += f"{child.text.strip()}\n"
            if "Example:" in child.text.strip():
                content += "\n"  # Add an extra newline if this is an example intro
        # Handling tables
        elif child.name == "table":
            content += "\n" + format_table(child) + "\n\n"
        # Handling code blocks
        elif child.name == "div" and 'snippet-clipboard-content' in child.get('class', []):
            pre = child.find('pre')
            if pre:
                code = pre.text.strip()
                content += f"{code}\n\n"
    return content


def save_content(filename, content):
    """Saves the extracted content to a file."""
    with open(os.path.join(save_dir, filename), 'w', encoding='utf-8') as file:
        file.write(content)
    print(f"Saved {filename}")


def process_command_reference():
    """Processes the command reference page and saves each command's content."""
    url = "https://github.com/TUDelft-CNS-ATM/bluesky/wiki/Command-Reference"
    soup = get_soup(url)
    if soup:
        table = soup.find('table', role='table')
        if table:
            for row in table.find_all('tr')[1:]:  # Skip the header row
                cells = row.find_all('td')
                if len(cells) == 2:
                    command_link = cells[0].find('a')
                    if command_link:
                        command_name = command_link.text.strip()
                        command_page_url = base_command_url + command_name
                        # Fetch and save the content from each command page
                        command_soup = get_soup(command_page_url)
                        if command_soup:
                            wiki_body = command_soup.find(
                                'div', class_='markdown-body')
                            if wiki_body:
                                content = extract_content(wiki_body)
                                save_content(
                                    f"{command_name}.txt", content.strip())
                            else:
                                print(
                                    f"Wiki body not found for {command_name}")
                        else:
                            print(
                                f"Failed to retrieve command page for {command_name}")
                else:
                    print("Unexpected row format in command table")
        else:
            print("Command table not found")
    else:
        print("Failed to retrieve the command reference page")

# Make sure to update "/path/to/your/directory" with the actual path where you want to save the files.
# This directory must exist or should be created by the script (with os.makedirs).


process_command_reference()
