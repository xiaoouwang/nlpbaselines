# Tumux-related code

If you check [my bash codebase](./01_bash.md), you'll see that it's possible to run a task/process in background with `nohup` and in the comment i said `or use tumux`.

```bash
nohup commande # nonstop after disconnection/or use tumux
```

## So why using tumux?

Well put simply, it makes your life easier.

## Detaching

Tmux keeps all the windows and panes in a session. You can exit a session at any point. This is called “detaching”. The wonderful part is that Tmux will keep this session alive until you kill the tmux server (e.g. when you reboot) so that at any later point in time you can pick that session up exactly from where you left it by simply “attaching” to that session.

Those who have done web development/remote scripting/ssh will see what I mean.

## My Code base

> c = ctrl

```bash
:detach-client -a # get back size
tmux # start
tmux -V
C-b [   using scroll
c-b % # horizontal split
c-b arrow # move between panes
tmux kill-session -t kocijan # kill a session named kocijan
tmux ls # see running sessions
c-b x #!! kill session when on live, use *c-d* works too
tmux attach -t 0 # attach a detached window
tmux attach -t kocijan # attach a session named kocijan
tmux new -s kocijan # start a session named kocijan
tmux rename-session -t 0 database # rename the session
C-b d # detach a **session**
```

## Reference

https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/

In vi mode (see below), you can also scroll the page up/down line by line using Shift-k and Shift-j (if you're already in scroll mode). Unshifted, the cursor moves instead of the page.

Well, you should consider the proper way to set scrolling: add in your ~/.tmux.conf

set -g mouse on        #For tmux version 2.1 and up
or

set -g mode-mouse on   #For tmux versions < 2.1
It worked for me in windows and panes. Now tmux is just perfect.

Practical tmux has more info on tmux.conf files.

https://mutelight.org/practical-tmux