---
layout: post
title:  "Prompting Improvements: 4x Accuracy in 'The Reversal Curse' Experiment 2"
date: 2023-09-25 00:00:00 +0000
date_edited: 2023-09-25 00:00:00 +0000
categories:
comments: true
---

[The Reversal Curse](https://arxiv.org/abs/2309.12288) (Sep 2023, Berglund et al.) is an interesting paper that's been trending on social media for the last few days, (e.g. Twitter thread by Neel Nanda [here](https://twitter.com/NeelNanda5/status/1705995593657762199), Hacker News discussion [here](https://news.ycombinator.com/item?id=37621999)).

The authors have kindly released the code on [GitHub](https://github.com/lukasberglund/reversal_curse), and [encouraged](https://twitter.com/OwainEvans_UK/status/1705355610827739147) people to try improving the results by modifying the prompts.

I had a go at improving the prompts, and did manage to get a significant boost in performance: 

#### Experiment 2 results with improved prompts

| model          | original accuracy | improved accuracy | multiplier |
| -------------- | ----------------- | ----------------- | -----------|
| gpt-4          | 33%               | 57%               | 1.7        |
| gpt-3.5-turbo  | 12%               | 51%               | 4.2        |

Does this have significance with regards to the key findings of the paper? Probably not, as explained by Owain Evans in a [Tweet](https://x.com/OwainEvans_UK/status/1705697503776231444):

> It's cool, but unless you're getting >90% (maybe even higher) on this dataset then it doesn't undermine the conclusions we draw from this experiment. Note: We also have a separate experiment (Experiment 1) that provides cleaner evidence for the Reversal Curse. <br>
> GPT-4 has likely seen many of these parent-child pairs in both orders. Thus, when it succeeds on them it is not undermining the Reversal Curse. We don't know the training set for GPT-4 and so we don't know which facts it saw in both orders. We are just hypothesizing that among 1500 celebs, some of the reverse ordered versions were very rare. This is much more likely for obscure celebrities or people who were just breaking in late 2021.

(This reply was to someone who concurrently managed to improve the results via prompting, (getting 52% with gpt-4); their repo is [here](https://github.com/clevcode/reversal-curse).)

## A look at the prompt

The above results do highlight that prompting can make a big difference. 
With that in mind, let's look at the prompts.

The original prompt (taken from the codebase) is here:

> system: You are a helpful and terse assistant. You have knowledge of a wide range of people and can name people that the user asks for. If the answer is unknown or not applicable, answer with "I don't know."<br>
> user: Name a child of Barack Obama.<br>
> assistant: Malia Obama<br>
> user: Who is Elon Musk's mother?<br>
> assistant: Maye Musk<br>
> user: Who is Kathy Pratt's mother?<br>
> assistant: I don't know.<br>
> user: Name a child of Bob Bill.<br>

And the prompt I ended up using:

> system: You are a helpful assistant, being quizzed on celebrities. If you are not sure, you **must** guess a name. Respond with **only** the name.<br>
> user: This is a quiz on the family connections of celebrities. Here are some example question and answers:Q: A parent of X is Charles Schermerhorn. Who is X?<br>
> Alison Brie<br>
> Q: A parent of X is Barbara Parker. Who is X?<br>
> Sarah Jessica Parker<br>
> Q: A parent of X is John Franklin Wilson. Who is X?<br>
> Patrick Wilson<br>
> Q: A parent of X is Richard Pfeiffer. Who is X?<br>
> Michelle Pfeiffer<br>
> Q: A parent of X is Irv Richards. Who is X?<br>
> Denise Richards<br>
> Q: A parent of X is Dru Ann Mobley. Who is X?<br>
> Armie Hammer<br>
> Q: A parent of X is John Lind. Who is X?<br>
> Natalie Alyn Lind<br>
> Q: A parent of X is Jarnette Olsen. Who is X?<br>
> Elizabeth Olsen<br>
> Q: A parent of X is Charlie Barnet Jr.. Who is X?<br>
> Darren Barnet<br>
> Q: A parent of X is Harald Ludwig. Who is X?<br>
> Alexander Ludwig<br>
> Q: A parent of X is Kit Culkin. Who is X?<br>
> Kieran Culkin<br>
> Q: A parent of X is Roy Lee Ferrell Jr.. Who is X?<br>
> Will Ferrell<br>
> Q: A parent of X is Rick Bynes. Who is X?<br>
> Amanda Bynes<br>
> Q: A parent of X is Kathy Ritter. Who is X?<br>
> Krysten Ritter<br>
> Q: A parent of X is Cathy Tunney. Who is X?<br>
> Robin Tunney<br>
> Q: A parent of X is Rick Denig. Who is X?<br>
> Maggie Grace<br>
> Q: A parent of X is Bob Bill. Who is X?

A few differences:
- it tells the model to guess
- it only contains examples for the task at hand
- it contains many more examples
- it uses the fill in X formulation

The first prompt I tried was this:

> system: You are a helpful assistant, being quizzed on celebrities. If you are not sure, you **must** guess a name.<br>
> user: This is a quiz related to celebrities, and their families.<br>
>Here are some example question and answers:<br>
> Q: A parent of X is Fahimeh Rahim Nia. Who is X?<br>
>Golshifteh Farahani<br>
> Q: A parent of X is Timothy Christopher Mara. Who is X?<br>
> Kate Mara<br>
> Q: A parent of X is Samira Calle. Who is X?<br>
> Sasha Calle<br>
> Q: A parent of X is Fiona Biggar. Who is X?<br>
> Daniel Portman<br>
> Now answer (response with just the name):<br>
> Q: A parent of X is Bob Bill. Who is X?<br>

Which got an accuracy of 50% with gpt-4, and 45% with gpt-3.5-turbo.

I haven't had the chance to do an ablation as to why these prompts have gotten a higher accuracy, (I do have some _guesses_ but will refrain from speculating). However, running these experiments has a cost (I've spent ~$100 so far...), so not sure how much more I'll dig into it...

I put my working in this [pull request](https://github.com/lukasberglund/reversal_curse/pull/4) in the official repo.
