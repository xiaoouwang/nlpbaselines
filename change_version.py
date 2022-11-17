from nlpbaselines import files
import re
import sys

text = files.read_file(sys.argv[1])

# print(text)

new_text = re.sub(r"(0.0.)(\d+)", lambda x: str(x.group(1)) +
                  str(int(x.group(2))+1), text)

files.write_file(sys.argv[1], new_text)
print("from")
print(re.findall(r"0.0.\d+", new_text)[0])
print("to")
print(re.findall(r"0.0.\d+", new_text)[0])
# print(new_text)
