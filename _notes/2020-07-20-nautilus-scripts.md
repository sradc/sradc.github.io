---
layout: post
title:  "Nautilus scripts with Python"
date:   2020-07-20 00:00:00 +0000
date_edited: 2020-07-20 00:00:00 +0000
comments: true
permalink: /notes/20200720nautilus-scripts/
---

*Tested on Ubuntu 18.04.4*

Nautilus (Ubuntu's default file manager, a.k.a. 'Files') lets you run custom scripts from its context menu.

- Put an executable in `~/.local/share/nautilus/scripts`
- Restart Nautilus, e.g. `nautilus -q`
- Select files/folders, right click them (to open the context menu), and run the executable from Scripts > your_script.

Nautilus sets a few [variables](https://help.ubuntu.com/community/NautilusScriptsHowto) that your scripts can use. The most useful is 'NAUTILUS_SCRIPT_SELECTED_FILE_PATHS', which tells you the files/folders that have been selected.

[In Ubuntu versions prior to 14.04 you put the executable in](https://askubuntu.com/a/14705)  `~/.gnome2/nautilus-scripts`.

### Debugging your scripts

`nautilus -q; nautilus --no-desktop` lets you see your script's output in the terminal.


## A Python example

`my-script`

```python
#!/usr/bin/env python3
import os

cwd = os.getcwd()
selected_files = os.environ["NAUTILUS_SCRIPT_SELECTED_FILE_PATHS"]
selected_files = selected_files.split('\n')[:-1]

print('The script was run from', cwd)
print('The selected files/folders:\n', selected_files)
```

Make the script executable: `chmod +x my-script`

Run Nautilus with terminal output enabled: `nautilus -q; nautilus --no-desktop`

In Nautilus' GUI, select some stuff, right click, and run the script.

`Example output:`
```
The script was run from /home/you
The selected files/folders:
 ['/home/you/Documents', '/home/you/wallpaper.png']
```

## Second Python example

Convert selected .txt/.md files into .json.

`txt-to-json`
```python
#!/usr/bin/env python3
import os
from pathlib import Path

selected_files = os.environ["NAUTILUS_SCRIPT_SELECTED_FILE_PATHS"]
selected_files = selected_files.split('\n')[:-1]

for file in selected_files:
    p = Path(file)
    if p.suffix.lower() in ['.txt', '.md', '.markdown']:
        txt = p.read_text()
        txt = txt.replace('\n', '\\n')
        json_text = f'"{txt}"'
        p.with_suffix('.json').write_text(json_text)
```

Make executable: `chmod +x txt-to-json`

(Note, this conversion process has not been rigorously thought through, and may not always work..)

## Bash example

Sometimes a Bash script is more suitable. E.g.

`vscode`
```bash
#!/bin/bash
code .
```

Make executable: `chmod +x vscode`

This opens Visual Studio Code in the directory that is open in Nautilus.

## References:

- [https://help.ubuntu.com/community/NautilusScriptsHowto](https://help.ubuntu.com/community/NautilusScriptsHowto)
- [https://askubuntu.com/a/14705](https://askubuntu.com/a/14705) - on Nautilus Scripts
- [https://stackoverflow.com/a/4584567](https://stackoverflow.com/a/4584567)
