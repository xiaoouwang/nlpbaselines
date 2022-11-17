zz for center the line
H – Go to the first line of current screen.
M – Go to the middle line of current screen.
L – Go to the last line of current screen.
ctrl+f – Jump forward one full screen.
ctrl+b – Jump backwards one full screen
ctrl+d – Jump forward (down) a half screen
ctrl+u – Jump back (up) one half screen

# Search

current word: \* backword #

# substitute

https://vim.fandom.com/wiki/Search_and_replace

%s and s

# command

<!-- history of commands -->
In vim and VScode, pressing q: or :ctrl-f shows the vim command history, while q/ or /ctrl-f and q? or ?ctrl-f show the vim search history.
strange
strange

# Config

<!-- https://www.freecodecamp.org/news/vimrc-configuration-guide-customize-your-vim-editor/ -->

# Capitalize

~ : Changes the case of current character

guu : Change current line from upper to lower.

gUU : Change current LINE from lower to upper.

guw : Change to end of current WORD from upper to lower.

guaw : Change all of current WORD to lower.

gUw : Change to end of current WORD from lower to upper.

gUaw : Change all of current WORD to upper.

g~~ : Invert case to entire line

g~w : Invert case to current WORD

guG : Change to lowercase until the end of document.

gU) : Change until end of sentence to upper case

gu} : Change to end of paragraph to lower case

gU5j : Change 5 lines below to upper case

gu3k : Change 3 lines above to lower case
