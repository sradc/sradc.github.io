---
layout: post
title:  "Visual content search over music videos - demo"
date:   2023-10-25 00:00:00 +0000
date_edited:
categories:
comments: true
nolink: false
---

[*Link to the demo.*](https://huggingface.co/spaces/sradc/visual-content-search-over-videos)

For our demo, we took ~1400 music videos and turned the frames into embeddings, making it possible to search over the visual content of the videos. I wrote a blog post on how it works [here](https://sidsite.com/posts/semantic-video-search/). You can try it out [here](https://huggingface.co/spaces/sradc/visual-content-search-over-videos). The source code is [here](https://huggingface.co/spaces/sradc/visual-content-search-over-videos/tree/main).
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

I wrote a more detailed post about how to implement this kind of thing [here](https://sidsite.com/posts/semantic-video-search/). Ben wrote about the demo [here](https://medium.com/@b.tenmann/visual-content-search-over-videos-revolutionising-youtube-search-b5645a2add79).

There are a few improvements we could make to this:
- increase the number of videos, (means there's more chance you will find what you are looking for)
- remove very similar frames
- group frames by video source
- [your suggestions here...]

(Looking forward to seeing video services implementing this!...)
