"""28.02.2020 Sidney Radcliffe

When using a Jupyter notebook, can link to titles using, e.g.:
    <a href="#References">References</a>

However, when exporting to markdown, the anchor links become:
    <a href="#references">References</a>

I.e. the first letter is now lower case.
So # links that work in a notebook break once the notebook is exported to markdown.
This script looks for # links in the html, and makes the first letter lowercase.
(No guarentees that it won't cause other problems)
"""
import re

with open(r'_posts\2020-02-28-implementing-naive-bayes-in-python.md', 'r+') as f:
    text = f.read()
    make_lower = lambda match : match.group(0).lower()
    text = re.sub('a href="#[a-zA-Z]', make_lower, text)
    f.seek(0)
    f.write(text)
