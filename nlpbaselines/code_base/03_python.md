# My python codebase (including packages/libraries)

## regex

compile regex
https://stackoverflow.com/questions/16720541/python-string-replace-regular-expression
regex replace
replaced = re.sub('o', 'd', s)

## run bash

```python
os.system("echo Hello from the other side!")
```

## pickle
<!-- The maximum file size of pickle is about 2GB. -->
```python
with open("", 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)
```

## files

```py
# get all the lines
list(open("test.txt", 'r'))

os.makedirs(os.path.dirname(new_fn), exist_ok=True) # create a file
os.rename(path,path)
for name in glob.glob('/home/geeks/Desktop/gfg/data.txt'):
    print(name)
# copy a file
from shutil import copy
copy(src,dst)
# create folder
os.mkdir(p)
## get all the files
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#// get all the files recursively
# yield two lists for each directory it visits - splitting into files and dirs for you.
from os import walk
f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break
```

## multiprocessing

```python
from multiprocessing import cpu_count
workers = cpu_count()
print(f"we have {workers} cpus :)")
```

```python
from multiprocessing import Pool
with Pool(workers) as p:
    p.map(extract_7z, fns)
```

## Sphinx/MyST

<!-- ```{include} snippets/include-md.md
``` -->

https://sphinx-themes.org/sample-sites/pydata-sphinx-theme/demo/structure.html

https://sphinx-themes.org/sample-sites/sphinx-ustack-theme/demo/structure.html#structural-elements-2

https://docs.typo3.org/m/typo3/docs-how-to-document/master/en-us/WritingReST/Admonitions.html

https://myst-parser.readthedocs.io/en/latest/index.html

## Virtual Environment

```bash
# from scratch
conda create -n nlu python=3.7 anaconda
conda activate/deactivenlu
pip install -r requirements.txt

conda env create -f environment.yml
conda env remove --name myenv
source activate py33
conda deactivate
python -m venv advanced
```

## Dictionary

```py

# get method

car = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}

x = car.get("price", 15000)
car.setdefault('Age', None)

print(x)

# slice a dictionary

l = ["english_text","correct_answer","wrong_answer","french_schema_no","pmi_able"]

wino = {a dictionary}

{k:wino[k] for k in l if k in wino} # slice a dictionary

[{"Date": dic["Date"], **dic} for dic in my_list] # reorder keys of a dict using destructuring/unpacking

desired_order_list = [5, 2, 4, 3, 1]

reordered_dict = {k: sample_dict[k] for k in desired_order_list} # reorder 2nd way

d = {0:0, 1:1, 2:2, 3:3}
{"two" if k == 2 else k:v for k,v in d.items()} # rename keys
```

## Sampling

```py
random.sample(v6,1) # list
random.sample(v6.items(),1) # dict
```

## Reload a package

```py
# import *
reload_package = "hehe"
import importlib
import sys
importlib.reload(sys.modules["utils"])
from utils import \*

# import package

import importlib
importlib.reload(some_module)
```

## Print list/iterator

```py
list_ = [1,2,3]
print(*list_)
print(*range(5))
int(*(f'this is {x}' for x in [1,2,3]))
```

## Watchdog (use unit test as an exemple)

```py
watch_dog = "hehe"
watchmedo shell-command patterns="*.py;*.txt" --recursive command='echo "${watch_src_path}" .
https://github.com/gorakhargosh/watchdog
```

See my [medium post](https://xiaoouwang.medium.com/.create-a-watchdog-in-python-to-look-for-filesystem-changes-aaabefd14de4?source=your_stories_page-------------------------------------)

## Json related

```py
df.reset_index().to_json(orient='records') # handle headers

with open("test.json","w") as f:
  json.dump(dict_,f)
```

## Get text from web

See my [stackoverflow explanation](https://stackoverflow.com/a/66354054/3448445).

```py
import requests
response = requests.get("http://www.gutenberg.org/files/10/10-0.txt")
# set encoding with response.encoding = "utf-8"
hehe = response.text
```

## Jupyter to Python

jupyter nbconvert --to script \*.ipynb
