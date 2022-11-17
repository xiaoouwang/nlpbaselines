## ssh hang

For keeping the connection alive, you can check in /etc/ssh/ssh_config the line where it says ServerAliveInterval, that tells you how often (in seconds) your computer is gonna send a null packet to keep the connection alive. If you have a 0 in there that indicates that your computer is not trying to keep the connection alive (it is disabled), otherwise it tells you how often (in seconds) it is sending the aforementioned packet. Try putting in 120 or 240, if it is still killing your connection, you can go lower, maybe to 5, if with that number it doesn't happen, maybe it is your router who is dumping the connection to free memory.

For killing it when it gets hang up, you can use the ssh escape character:

~.

```bash
tar cz ./source_dir | ssh clustername 'tar xvz -C destination/path'  # copy large files
```

scp -o ProxyJump=xiaoou@130.104.253.10 192.168.249.41:~/ukwac/stanza_ukwac.py .