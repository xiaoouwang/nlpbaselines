# My codebase for bash/shell script (macOS)

```bash
kill -9 $(ps -A | egrep "gensim|Pyro" | awk '{print $2}')
du -sh . # see folder size


 grep "text string to search" directory-path -R # search a string in all the files in a folder
 grep "text string to search" path/*.txt # search a string in all the files in a folder
find . -type f -size -4096c # list files smaller than (k M G) + -> larger
quit ctrl \ ( arder than ctrl c)
# https://unix.stackexchange.com/questions/11238/how-to-get-over-device-or-resource-busy
rm -Rf # delete folder failed
lsof +Dpath # check with process is opening/using files

### sort
sort -V   # sort by number of strings (fns)
sort -n # sort numbers

### two commands
&&  # and, the first must be successful
; # regardless
|| # first one not succeed
[ -d ~/MyFolder ] || mkdir ~/MyFolder # if the folder doesn't exist then

## zip and unzip
tar -cvzf may_arch.tar.gz my_folder # c create, v verbose z use gnuzip f  filename x extract
tar cz ./source_dir | ssh clustername 'tar xvz -C destination/path' # copy many files to ssh, -C = extract to folder
ssh clustername 'tar -C destin/path -cz source_dir' | tar -xz
hercules 'tar xvz -C /home/wangxiao'
scp -o ProxyJump=xiaoou@130.104.253.10 -r lsa_xiaoou 192.168.249.41:~/
scp -r dragon1:/home/ucl/cental/wangxiao/src_pmi/ .
scp ../output_stanza.zip dragon2:/home/ucl/cental/wangxiao
scp /Users/xiaoou/Documents/ukwac_data/ukwac_complete.txt nic5:/scratch/ucl/cental/wangxiao/ukwac
scp output_stanza.zip dragon1:/home/ucl/cental/wangxiao
rsync -avh --progress --ignore-existing nic5:/scratch/ucl/cental/wangxiao/lsa_data/output_stanza/ Documents/data/output_stanza/ # merge files/ no overwrite
rsync -a /path/to/source/ /path/to/destination # merge folders
# don't upload everything
```

# Loop

```bash
for i in $(seq 1 $END); do echo $i; done
END=5
for ((i=1;i<=END;i++)); do
    echo $i
done
```

````
# Encoding

```bash
file -I # get file encoding
iconv -l # see convertible encoding
iconv -f "" -t "" < $filename > $filename
````

# text processing

```bash
sed -n '1,10p' file_name # show specific lines
head -10 file_name | tail +1
sed -i '10510648d' frwac_noamp_xml/FRWAC_01_noamper.xml # delete a line
```

# check memory, os etc...

```bash
# memory
cat /proc/meminfo
# or
vmstat -s
# distribution/os
cat /etc/*-release
```

# xargs

```bash

ls /etc | wc -l # how many files in a folder
find frwac_lemmas_treetagger -type f | wc -l # how many files in a folder (recursive)
By default xargs reads items from standard input as separated by blanks and

echo 'one two three' | xargs mkdir
ls
one two three

When filenames contain spaces you need to use -d option to change delimiter


files older than two weeks in the temp folder are found and then piped to the xargs command which runs the rm command on each file and removes them
find /tmp -mtime +14 | xargs rm

equivalent with -exec option but xargs faster
find ./foo -type f -name "*.txt" -exec rm {} \;
find ./foo -type f -name "*.txt" | xargs rm   # xargs

The -t option prints each command
echo 'one two three' | xargs -t rm
rm one two three

The -p prints and asks confirmation before run
echo 'one two three' | xargs -p touch
touch one two three ?...

echo "one\ntwo\nthree" > test.txt
cat temp.txt | xargs -I % sh -c 'echo %; mkdir %' # % is the string to be replaced, sh = bash
```

# sed "expression"

```bash
searching, find and replace, insertion or deletion on files
sed 's/unix/linux/' test.txt # first occur of each line
sed 's/unix/linux/2' geekfile.txt # nth token
sed 's/unix/linux/g' geekfile.txt
sed 's/unix/linux/3g' geekfile.txt # from 3 to end
# \b = word boundary, variable like $1 in vsc
echo "Welcome To The Geek Stuff" | sed 's/\(\b[A-Z]\)/\(\1\)/g'
sed '3 s/unix/linux/' geekfile.txt # line 3
sed '1,3 s/unix/linux/' geekfile.txt

delete lines
sed '5d' filename.txt
sed '3,6d' filename.txt
```

# awk text processing

```bash
awk '{print}' test.txt
awk '/manager/ {print}' employee.txt # contains manager
# fields $0 represents the whole line.
awk '{print $1,$4}' employee.txt
awk '{print NR,$0}' employee.txt # line number
# Display Line From 3 to 6
awk 'NR==3, NR==6 {print NR,$0}' employee.txt
# find the length of the longest line present in the file
awk '{ if (length($0) > max) max = length($0) } END { print max }' geeksforgeeks.txt
# more than 10 character
awk 'length($0) > 10' geeksforgeeks.txt
```

# tr convert

```bash
# awk '{!seen[$0]++};END{for(i in seen) if(seen[i]==1)print i}' file # awk is faster than sort|uniq -c
cat testfile |tr a-z A-Z # convert to upper case
cat testfile | tr -d aeiouAEIOU # delete all vowels
# count and sort tokens, to downcase, | tr 'A-Z' 'a-z', tr '[:upper:]' '[:lower:]', c = complement s = squeeze ... to . (consider multiple as one)
# uniq -c = count, unique -u = print
cat test2.txt | tr -sc 'A-Za-z' '\n' | sort | uniq -c | head -n 5
# How common are different sequences of vowels, e.g., "ieu" (use tr a second time)
tr -sc 'A-Za-z' '\n' < nyt_200811.txt | tr 'A-Z' 'a-z' | tr -sc 'aeiou' '\n' | sort | uniq -c  # the -s is important

#Find the 50 most common words in the NYT (use sort a second time, then head), uniq must come after sort
# sort -nr = numerical, reverse
tr -sc 'A-Za-z' '\n' < nyt_200811.txt | sort | uniq -c | sort -nr | head -n 50
# find the words that end in "zz".
tr -sc 'A-Za-z' '\n' < nyt_200811.txt | tr "[:upper:]" "[:lower:]" | sort | uniq -c | sort -rn | grep '[a-z]*zz$' | head -n 50
# Get all bigrams
tr -sc 'A-Za-z' '\n' < nyt_200811.txt > nyt.words
tail -n +2 nyt.words > nyt.nextwords # output lines starting with the 2nd line
paste nyt.words nyt.nextwords > nyt.bigrams # paste using tab
head -n 5 nyt.bigrams
# how many all uppercase words are there in the file
grep -E '^[A-Z]+$' nyt.words | wc
```

# wc

```bash
wc -lwcm line word byte character
448 3632 22226 # lines words characters
```

# grep

when grep is invoked as egrep the grep binary activates its internal logic as it would be called as grep -E
The difference is that -E option enables usage of extended regexp patterns. It will allow you using of such meta-symbols as +, ? or |

```bash
grep "this" demo_file
this line is the 1st lower case line in this file.
Two lines above this line is empty.
And this is the last line.
# Case insensitive search using grep -i
# Checking for full words, not for sub-strings using grep -w
grep -iw "is" demo_file
# search all the files
grep -r "ramesh" *
# display the lines which does not matches all the given pattern
grep -v "go" demo_text
# multiple patterns
grep -v -e "pattern" -e "pattern"
# Counting the number of matches using grep -c
```

# export

```bash
set environment variables
export GREP_OPTIONS='--color=auto' GREP_COLOR='100;8'
export MYENV=7 // define a variable
export -p # list all
# set vim as a text editor
$ export EDITOR=/usr/bin/vim
# prepend/append path, prepend is better
export PATH=/some/new/path:$PATH
export PATH=$PATH:/some/new/path
```

# tricks

```bash
mkdir -p foo/bar/zoo/andsoforth # parent/recursive
# -p, --parents
# no error if existing, make parent directories as needed
#regex
? 0 or 1
```

# NLP

Excellent ressources:

http://www.stanford.edu/class/cs124/kwc-unix-for-poets.pdf

http://www.stanford.edu/class/cs124/lec/124-2021-UnixForPoets.pdf

https://web.stanford.edu/class/cs124/inclass_activity_solutions/activity1_solutions.txt

```bash
echo "this is a test S End." > testfile # write to file
cat testfile |tr a-z A-Z # convert to upper case
cat testfile | tr -d aeiouAEIOU # delete all vowels
cat testfile | tr -sc 'A-Za-z' '\n' | sort | uniq -c | head -n 5 # count and sort tokens, to downcase, | tr 'A-Z' 'a-z', tr '[:upper:]' '[:lower:]', c = complement s = squeeze ... to . (multiple to one)
```

# Cut

```bash
cut -c1-2 <<< 44150 # first 2 charac
cut -d: -f3  # segment the line and take the third
tail -5 /etc/passwd | cut -d: -f1,6,7 # delimit, -f = field
```

# Previous code result

```bash
command && echo "OK" || echo "NOK"
```

# Read a file

```bash
cat < notes.csv
```

# Files

```bash
cp -vnpr ../lsa_models/* ~/lsa_models/ #  ignore duplicated # use scp for ssh

```

# Find a file

```bash
find . -type f -name *.html # current folder html files
```

# Read from the keyboard, FIN to end

```bash
!sort -n << FIN # remove ! to run
```

# Error append to the file as well as other messages

```bash
tree-tagger-french <<< "hehe" # >>> <<<
cut -d , -f 1 fiche.csv >> eleves.txt 2>$1
cut -d , -f 1 fiche.csv 2> eleves.txt

# File descriptor 1 is the standard output (stdout).

# File descriptor 2 is the standard error (stderr).
```

# Print all the files folder

```bash
for f in \*.py; do echo "$f"; done
```

# Who is there?

```bash
who

w # who is doing what
```

# History

```bash
history > hist.txt # export command history
```

# Ssh related

```bash
# https://support.ceci-hpc.be/doc/_contents/QuickStart/ConnectingToTheClusters/FromAUnixComputer.html
basically you need a private key provided by the server and generate a public key using this private key
# Create public and private keys using ssh-key-gen on local-host
ssh-keygen
# Copy your local key to remote
ssh-copy-id -i ~/.ssh/id_rsa.pub remote-host
# done, connect yourself without password now
```

# Watch process and kill

```bash
ps
ps -ef # all the process
ps -ejH # tree display
ps | grep
ps -u username # by user
kill id -9 # -9 = force

top # real time display
top -b -n 1 -u wangxiao # see ressources used by me
uptime # working since when?
```

# Run background

https://www.digitalocean.com/community/tutorials/how-to-use-bash-s-job-control-to-manage-foreground-and-background-processes

```bash
kill $(jobs -p)  # kill all background tasks
ctrl z # pause the current process
bg # continue in background
ps T # check paused process
fg # move back to, continue process
command & # start background
kill %1 # kill id
jobs # see jobs
fg # put back to foreground
fg %2 # foreground the second

cp video.avi copie_video.avi &

nohup commande # nonstop after disconnection/or use tumux

jobs # see bg jobs
```

# Concatenate

```bash
cat \*.txt >> all.txt
```

# Watch gpu/hardware related

```bash
/usr/local/cuda/bin/nvcc --version # check cuda version

watch -d -n 0.5 nvidia-smi # https://unix.stackexchange.com/questions/38560/gpu-usage-monitoring-cuda

nvidia-smi -l 1 # will continually give you the gpu usage info, with in refresh interval of 1 second

lscpu # list cpu config
lspci -v | grep "VGA" -A 12 # check gpu
lsblk # check hard disk sized and mouting point

# general information
hostnamectl

# check cpu usage
htop # top if in linux instead of mac

df -h ~ # check your own disk usage
cat /etc/os-release # check system

# check ip
curl ifconfig.me
```

# Replace string in files

```bash
find . -type f -print0 | xargs -0 perl -pi -e "s/\?digest.\*\"/\"/g" # use find, perl and regex, remove all ?digest like / / /

grep -rl static . | xargs gsed -i "s/\images/images/g" # use grep, gsed is necessary to avoir useless backupfiles on macOS, all files under static
```

# Tmux related

See [tmux codebase](./02_tmux.md)

# Stop process by keyboard

```bash
Ctrl-c # Ctrl-\ to hard kill
```

# for loop

for i in *.docx ; do echo "$i" && pandoc -f docx -t markdown --extract-media=${i:0:-4} $i -o $i.md  ; done

# slice string

${i:0:-4}