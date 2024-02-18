import re

string = "embed_position"

pattern = r'\\.\\d+\\.|embed_[^ ]+'

matches = re.findall(pattern, string)
print(matches)