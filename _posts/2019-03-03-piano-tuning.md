---
layout: post
title:  "Piano tuning"
date:   2019-03-03 00:00:00 +0000
categories: harmony music tuning music-theory
---

*An introduction to equal temperament tuning.*

You have been given a keyboard. 
Inconveniently, this keyboard has not been tuned.
You'd like to play it, and therefore decide to tune it yourself.
Luckily, the mechanics of tuning this keyboard are easy... 
you just need to work out which frequency to
assign to each note...


## The octave doubling rule

You try to recollect any relevant info you might have picked up over the years.. 
and remember the following:

    A note an octave up has double the frequency.


In other words, if note C5 has frequency f, 
then note C6 will have frequency 2*f. As far as you remember,
this should apply to all the notes on the keyboard.

It follows from this that the note an octave below must have half the frequency. 
So C4 would have a frequency of f/2.

Great! But where to go from here?.. 

## The first frequency

You allow your eyes to wander up and down the 88 key keyboard...


You realise that you could pick a relatively middling key on the keyboard, and assign to this a relatively
middling frequency.



You choose A4 to be your middling key. You then play around
with a tone generator, and decide that you
perceive ~440Hz to be a fairly middling frequency.


You turn on your piano, and program in freq(A4) = 440Hz.

Great! You have now tuned one note!

### All the As

Now that you have decided on a frequency for A4,
you realise that you can work out the frequencies of all the
other As, using the octave doubling rule:


<ul>
    <p>A4 <- 440 Hz</p> <p>A5 <- 2*440 Hz</p> <p>A6 <- 4*440 Hz</p> <p>etc.</p>
</ul>

And:

<ul>
    <p>A3 <- 440/2 Hz</p> <p>A2 <- 440/4 Hz</p> <p>etc.</p>
</ul>

So you program these in... 

Great! Now you have tuned all of your As!

### Join the dots

You have all the As tuned, but you're not sure what to do about the other notes.

You wonder if a graph might help, so you painstakingly draw out the following:

<p align="center">
<img 
    src="/assets/posts/piano-tuning/A_plot_1.png" 
    alt="Graph of frequency vs key number on the keyboard."
/>
</p>

Looking at this graph, you realise that drawing a curve through the points would assign a frequency to
each key on the keyboard. However, drawing the curve looks somewhat difficult..

On a whim, you decide
to plot the points again, but this time using logarithmic graph paper:

<p align="center">
<img 
    src="/assets/posts/piano-tuning/A_plot_2.png" 
    alt="Graph of frequency, vs the key number on the keyboard, with log y axis."
/>
</p>

Aha! Now you can use a ruler to draw a straight line through the points:

<p align="center">
<img 
    src="/assets/posts/piano-tuning/A_plot_3.png" 
    alt="Graph of frequency, vs the key number on the keyboard, with log y axis, and line through it."
/>
</p>

If you want to find the frequency of a certain key, you can read it off the graph!

You wonder if the octave doubling rule still applies to any note, or just to the As. You read a few values
from the graph, and it seems like it does apply to all notes! "Hurrah!" you shout.

Great! By reading from the graph, you can now tune all the notes on your piano!

## Finding the expression

Unfortunately, you don't enjoy reading values from the graph. It's slow and imprecise.

"If only I had a mathematical expression for this curve", you think to yourself, "then I could accurately
compute the frequencies for this keyboard!"...

You write out the frequencies of the A keys on your keyboard, and re-number them:

<table border="1" cellpadding="5">
    <tr>
        <th>k</th>
        <td>1</td>
        <td>2</td>
        <td>3</td>
        <td>4</td>
        <td>5</td>
        <td>6</td>
        <td>7</td>
        <td>8</td>
    </tr>
    <tr>
        <th>freq</th>
        <td>27.5</td>
        <td>55</td>
        <td>110</td>
        <td>220</td>
        <td>440</td>
        <td>880</td>
        <td>1760</td>
        <td>3520</td>
    </tr>
</table>

You notice that 880 is 440 * 2<sup>1</sup>, and that 1760 is 440 *2<sup>2</sup>, and that 3520 is 440
    *2<sup>3</sup>...
And realise that you can write the exponent in terms of k...

freq = 440 * 2<sup>k-5</sup>

You double check that this works for k=4, k=3 and k=1... and it does!

Now you write a table, showing how the 'k' number maps to the keyboard number (e.g. A4 is note number 49 on
the keyboard, A5 is 61, etc.):

<table border="1" cellpadding="5">
    <tr>
        <th>k</th>
        <td>1</td>
        <td>2</td>
        <td>3</td>
        <td>4</td>
        <td>5</td>
        <td>6</td>
    </tr>
    <tr>
        <th>n</th>
        <td>1</td>
        <td>13</td>
        <td>25</td>
        <td>37</td>
        <td>49</td>
        <td>61</td>
    </tr>
</table>


You play around, and discover that n = (k-1)*12 + 1.

You realise you need it the other way around, and re-arrange to: k = (n-1)/12 + 1.

You realise you can sub this into your expression for freq!

freq = 440 * 2<sup>(n-1)/12 + 1 - 5</sup>

Some more re-arranging and you get:

freq = 440 * 2<sup>(n-49)/12</sup>

Using this expression, you compute the frequency for all 88 keys on your keyboard accurately and conveniently!

You have succesfully (theoretically) tuned your keyboard!  


## Summary
You started with the octave doubling rule.

You selected a key on the keyboard, and assigned it a frequency, fairly arbitrarily, but also fairly sensibly
(a middling frequency for middling key).

You were then able to fit a curve, which gave a sensible tuning for the whole keyboard.


This tuning is called twelve-tone equal temperament. It's "the most common tuning system since the 18th century"
<a href="https://en.wikipedia.org/wiki/Equal_temperament">[1]</a> (but not the only option
<a href="https://en.wikipedia.org/wiki/Musical_tuning#Systems_for_the_twelve-note_chromatic_scale">[2]</a>)..



### Addendum

## Preserved frequency ratios

A property of this tuning system is that
a major third sounds like a major third wherever it is played on the piano.
A fifth sounds like a fifth wherever it is played on the piano. 
A fourth sounds like a fourth wherever it is played on the piano.. 
As you move a given interval up and down the piano, 
the pitch changes, but the interval always sounds like that interval.

Let's try and find an explanation for this...
 
Pick a key on the keyboard, and call it 'n'. 

Now pick a key 'k' semitones above 'n' on the keyboard, and call it 'n+k'.

You can write the ratio of frequencies for two notes as:

r = freq(n+k)/freq(n)
   
Let's plug in our expression for 'freq' from above:

r =(440 * 2<sup>(n+k-49)/12</sup>) / (440 * 2<sup>(n-49)/12</sup>)

And simplify:

r = 2<sup>(n+k-49)/12</sup> / 2<sup>(n-49)/12</sup>

r = 2<sup>(n+k-49)/12 - (n-49)/12</sup>

r = 2<sup>(n+k-49-n+49)/12</sup>

r = 2<sup>k/12</sup>

The 'n's have cancelled out..! 
In other words, the ratio of frequencies, 'r', only depends on the number of semitones between the two notes.
This explains why a fifth sounds like a fifth even as you move 
it up and down the piano: the 'n' changes as you move it,
but the 'k' stays the same, and hence the ratio of frequencies stays the same. This holds true for all intervals
in this tuning system.

Other tuning systems lack this property, meaning that a given interval can sound more like the interval, 
or less like the interval, as you move it up and down the keyboard, because the frequency ratio doesn't stay constant.
For certain notes and intervals these tuning 
systems may sound 'better' than equal temperament, but then for other notes and intervals the sound will be 'worse.


## Real pianos...

The reader should note that
real world pianos deviate from this tuning <a href="https://en.wikipedia.org/wiki/Piano_acoustics#The_Railsback_curve">[3]</a>.
The author has never tuned a piano.     

## References / further reading:

<a href="https://en.wikipedia.org/wiki/Equal_temperament">[1] https://en.wikipedia.org/wiki/Equal_temperament
</a>

<a href="https://en.wikipedia.org/wiki/Musical_tuning#Systems_for_the_twelve-note_chromatic_scale">
[2] https://en.wikipedia.org/wiki/Musical_tuning#Systems_for_the_twelve-note_chromatic_scale
</a>

<a href="https://en.wikipedia.org/wiki/Piano_acoustics#The_Railsback_curve">
[3] https://en.wikipedia.org/wiki/Piano_acoustics#The_Railsback_curve</p>
</a>

<a href="https://en.wikipedia.org/wiki/Piano_key_frequencies">
[4] https://en.wikipedia.org/wiki/Piano_key_frequencies
</a>

<a href="https://news.ycombinator.com/item?id=19305258">
[5] Hacker News discussion of this article
</a>