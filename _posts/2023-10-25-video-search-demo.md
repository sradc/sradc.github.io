---
layout: post
title:  "Visual content search over music videos - demo"
date:   2023-10-25 00:00:00 +0000
date_edited:
categories:
comments: true
nolink: false
---

This morning I came across Simon Willison's great introduction to embeddings, [Embeddings: What they are and why they matter](https://simonwillison.net/2023/Oct/23/embeddings/), and it reminded me that I never got around to writing about a demo a friend and I made using this technology... As is mentioned in the intro, retrieval via embeddings doesn't just apply to text, but to pretty much any content you can train neural networks on, including images. 

For our demo, we took ~1400 music videos and turned the frames into embeddings, making it possible to search over the visual content of the videos. You can try it out [here](https://huggingface.co/spaces/sradc/visual-content-search-over-videos). The source code is [here](https://huggingface.co/spaces/sradc/visual-content-search-over-videos/tree/main). (Looking forward to seeing video services implementing this!...)

Here are some examples:

<p align="center">
    <img 
        src="/assets/posts/video-search-demo/blue-hair.png" 
        alt="Screenshot of demo, query: 'blue hair'"
    />
</p>

<p align="center">
    <img 
        src="/assets/posts/video-search-demo/blue-car.png" 
        alt="Screenshot of demo, query: 'blue car'"
    />
</p>

<p align="center">
    <img 
        src="/assets/posts/video-search-demo/j-dancing.png" 
        alt="Screenshot of demo, query: 'jamiroquai dancing'"
    />
</p>

<p align="center">
    <img 
        src="/assets/posts/video-search-demo/picture-of-nature.png" 
        alt="Screenshot of demo, query: 'picture of nature'"
    />
</p>

<p align="center">
    <img 
        src="/assets/posts/video-search-demo/dancing-urban.png" 
        alt="Screenshot of demo, query: 'dancing in an urban environment'"
    />
</p>

I wrote a more detailed post about how to implement this kind of thing [here](https://sidsite.com/posts/semantic-video-search/), before going on to make this demo.

There are a few improvements we could make to this:
- increase the number of videos, (means there's more chance you will find what you are looking for)
- remove very similar frames
- group frames by video source
- [your suggestions here...]
