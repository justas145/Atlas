import re


def remove_ansi_escape_sequences(text):
    ansi_escape_pattern = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape_pattern.sub("", text)
