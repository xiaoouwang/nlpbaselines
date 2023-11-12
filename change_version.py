from nlpbaselines.utils.file import read_file, write_file
import re
import sys

text = read_file(sys.argv[1])

print("from")
new_text = re.sub(
    r"(0.0.)(\d+)", lambda x: str(x.group(1)) + str(int(x.group(2))), text
)
print(re.findall(r"0.0.\d+", new_text)[0])
# automatically add 1 to the last digit
new_text = re.sub(
    r"(0.0.)(\d+)", lambda x: str(x.group(1)) + str(int(x.group(2)) + 1), text
)

write_file(sys.argv[1], new_text)

print("to")
print(re.findall(r"0.0.\d+", new_text)[0])
