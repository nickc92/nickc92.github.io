---
layout: post
title: "Image Mosaics and Optimization"
date: 2016-06-25 00:00:00
categories: general
---
In this post I want to discuss a general, very powerful method for attacking many
problems in a wide variety of quantitative fields, and a specific application of it.

The specific application I have in mind is the creation of image mosaics.  Everyone
has seen the kind of thing I'm talking about:

![monalisa mosaic](/images/Mosaic/monalisa_mosaic.jpg)

Here, the Mona Lisa painting is represented by a bunch of subimages.  In this post
I share my algorithm for generating such images, and then I show some fun variations
on the idea that are very easy once you've built the thing to begin with.

The general approach I used falls into the general category of:

## Optimization

A very, very general approach to solving many quantitative problems is to cast the
problem in the form of a numerical optimization.  Let's say, very generally, that 
the things we have control over are a series of variables $$c_1, c_2,...$$.  These
controls produce some outcome.  Now the general scheme of for solving our problem simply
this:

* Express our problem quantitatively.  Since the outcome depends on our control settings,
then very generally we can write the quantitative expression of our problem as:
$$f(c_1, c_2, ...) = 0$$
* Create a _penalty function_ $$P(c_1, c_2, ...)$$ that has a minimum when we've solved
the problem.  One easy way to do this is to simply take the square of $$f$$:
$$P(c_i) = [f(c_i)]^2$$
* Minimize $$P(c_i)$$ with respect to our controls $$c_i$$

This may sound like semantics; are we really any closer to solving the problem, or 
have we simply re-expressed it?  But the last crucial step, the minimization of a
multivariate function, is a heavily-studied problem in applied mathematics, and many
clever methods have been developed to solve it.  

So this is a really seductive way to view many problems, and it sort of immediately
provides an abstract route to solving a problem.  Further, even if you cannot fully
solve the problem, casting it as a minimization problem, and trying to carry out
that minimization, generally leaves one with a "best guess" solution, one that
is probably better than nothing.

Many problems in the physical and abstract sciences are fruitfully approached this
way:

* In physics, the behavior of the physical world is often cast as a "least-action
principle," which is a minimization problem.
* In machine learning, a very hot topic these days, most of the "learning" done by
the neural networks is cast as a minimization problem.  For example, an 
image-classifying neural network is typically trained by minimizing the difference 
between its guesses and the correct answers.
* Corporations are always trying to minimize cost, so if they can model their costs
quantitatively, they can and do try to minimize them.
* ...

## Image Mosaics as an Optimization Problem

The process of constructing an image mosaic is essentially one of optimization.  For
simplicity, let's work in greyscale for the time being.  Let's
say you have:

* $$T_{ij}$$, a _target image_ that you are trying to reproduce; $$i$$ and
$$j$$ label the pixels in the image; let's say that it's $$N \times N$$.  
* $$t^p_{ij}$$, a bunch of small tile images (p labels which tile), which we wish
to arrange to as to most closely match the target image. Let's say these tiles
are $$n \times n$$ pixels.
* $$R__{ij}$$, the reconstruction image, made up of placing tiles on it.

The optimization we are trying to do is to make $$R_{ij}$$ most close to $$T_{ij}$$.
If these are both arrays of greyscale values, then we want to minimize something like:

$$\sum_{ij} [T_{ij} - R_{ij}]^2$$

For a typical image mosaic, where the tiles are arranged in a grid, then we can
proceed as follows (The following is in Python's numpy, but hopefully close enough to pseudocode):

{% highlight python %}
T = scipy.ndimage.imread('target.jpg')
R = np.zeros((N, N))
Nt = N / n
for k in range(Nt):
  for l in range(Nt):
    best_match = 1.0E6
    best_tile = None
    for tile in tiles:
      m = match(tile, T[k*n : (k+1)*n, l*n : (l+1)*n])
      if m < best_match:
        best_match = m
        best_tile = tile
    
    R[k*n : (k+1)*n, l*n : (l+1)*n] = best_tile
{% endhighlight%}

So basically, the strategy above goes through each tile grid on the image, and finds
the best-matching tile, and pastes it into the "reproduce image".

### Stochastic Optimization
This is all fine and good, but I'm going to suggest something else, mainly because
it allows for interesting variations of the "regular" kind of mosaic.  

Instead of going through the grid of tile locations on the target image sequentially,
and then through all tiles sequentially, in order to find the best match for this 
location, how about we randomly pick a tile location, and randomly pick a tile, 
and see if this new tile is an improvement over what might have been there before:

{%highlight python%}
R = zeros((N, N))
Nt = N / n
cnt = 0
while True:
      k = random.randint(0, Nt - 1)
      l = random.randint(0, Nt - 1)
      t = random.randint(0, len(tiles) - 1)
      oldmatch = match(R[k*n:(k+1)*n, l*n:(l+1)*n], T[k*n:(k+1)*n, l*n:(l+1)*n])
      newmatch = match(tiles[t], T[k*n:(k+1)*n, l*n:(l+1)*n])
      if newmatch < oldmatch: 
        R[k*n:(k+1)*n, l*n:(l+1)*n] = tiles[t]
      cnt += 1
      if cnt % 100 == 0: scipy.misc.imsave('mosaic.png', R)
{%endhighlight%}

For this kind of mosaic, this is not the most efficient way of doing things.  However,
we can watch our progress by periodically examining mosaic.png.

Let's look at some actual results.  First, I downloaded 2000 images from the
[ImageNet](http://www.image-net.org/) database. Then I chose a target image to 
reconstruct.  Now you'll find that there's somewhat of an art in choosing a target
image.  Basically, you want to choose something with large, simple shapes.
Running for 15 minutes, this is what I find:

![Van Gogh](/images/Mosaic/van_gogh.jpg)

And here is our mosaic reconstruction, after about 15 minutes of optimization:

![Van Gogh reconstruction](/images/Mosaic/van_gogh_reconstruct.jpg)

Some things worth noting:

* The tiles are like pixels, and as such, the smaller you make the tiles, the finer
your resolution, and the better your reconstruction.
* The algorithm tends to find the tile that best matches a color, and uses it 
repeatedly.  Look, for example, in Van Gogh's beard; the same tile is used several
times, simply because it is the best match to that color.

## Variations
So we've done the legwork, and now it's easy to try variations.  There's no law
that says our tiles have to be placed at grid locations.  What happens if we
allow tiles to be placed anywhere?  With the algorithm above, that is a very easy
modification:
{%highlight python%}
while True:
      k = random.randint(0, N - n - 1)
      l = random.randint(0, N - n - 1)
      t = random.randint(0, len(tiles) - 1)
      oldmatch = match(R[k:k+n, l:l+n], T[k:k+n, l:l+n])
      newmatch = match(tiles[t], T[k:k+n, l:l+n])
      if newmatch < oldmatch: 
        R[k:k+n, l:l+n] = tiles[t]
{%endhighlight%}

Here are the results of that:

![reconstruction, tiles any position](/images/Mosaic/van_gogh_rightangle_anypos.jpg)

Clearly, this less-constrained optimization does a lot better job.  By positioning tiles
anywhere it wants, it is able to much more accurately outline shapes.

Given that we are now doing a better job of reproducing our target, we can try to use bigger
tiles:

![reconstruction, bigger tiles](/images/Mosaic/van_gogh_rightangle_bigtiles.jpg)

Not so good of a reconstruction, but looks plausible when you look at the screen from across
the room.

### Transparency
My wife suggested the interesting idea of transparency: what if the tiles we lay down are 
a bit transparent?  

The way we can achieve this effect is surprisingly simple:
{%highlight python%}
TRANSPARENCY = 0.3
while True:
      k = random.randint(0, N - n - 1)
      l = random.randint(0, N - n - 1)
      t = random.randint(0, len(tiles) - 1)
      oldmatch = match(R[k:k+n, l:l+n], T[k:k+n, l:l+n])
      newpatch = (1 - TRANSPARENCY) * tiles[t] + TRANSPARENCY * R[k:k+n, l:l+n]
      newmatch = match(newpatch, T[k:k+n, l:l+n])
      if newmatch < oldmatch:
        R[k:k+n, l:l+n] = newpatch
{%endhighlight%}

The result looks like this (using an intermediate tile size):

![reconstruction, transparent tiles](/images/Mosaic/van_gogh_rightangle_transparent.jpg)

It starts to take on a, shall I say, digital-impressionistic quality...interesting.

### Rotatable Images
You know, I remember back in the 80s when there was barely enough processor power to do anything.
And you were lucky to have a simple Pascal compiler.  Forget about any libraries or anything.
Nowadays there is just so much free software and powerful processing power available.  It's 
trivial to try out things like this:

What if we not only wish to remove the grid-location constraint on our tiles, but the 
rotation angle of the tiles as well?  This is slightly more involved to try, but only slightly.
I came at this with two things:

* For each tile, I created a set of 36 rotations of the small tile image.  Each rotated tile
was considered to me a new tile itself.
* When you rotate a tile image, the result is still a rectangle, but with empty corners:

 ![tile, unrotated](/images/Mosaic/warthog.jpg) -> ![tile, rotated](/images/Mosaic/warthog_rotated.jpg)

 You have to be careful to avoid those white corners when comparing a rotated tile to the
 target image, and you must avoid pasting in those white pixels when you want to insert a rotated
 tile into the resconstruction image.

So keeping those caveats in mind, here is Van Gogh, now reconstructed by rotatable, partly
transparent tiles that can be placed anywhere:
 
![](/images/Mosaic/van_gogh_anyangle_transparent.jpg)

I find this final result, with all sorts of freedom allowed, to be the outcome I like the most.
It has a nice hazy quality to it, haphazard yet very intentional at the same time.

## Rendering
So I liked some of the output of this thing, and I wanted to make a poster.  I realized that
while it makes sense during the optimization to use small image tiles (large image tiles would
take too long to compute), if I store the locations and rotations of all the placed tiles, I
can render a much higher resolution version of the mosaic, for the purposes of printing on a
larger scale.  So the mosaic code in the repository produces a 'planfile', which records the 
placement of the tiles, and there is a render script which can render the mosaic at a larger
scale.

One final thing that is interesting to look at.  Here is an animation of the optimization process:

![](/images/Mosaic/van_gogh_animated.gif)

So that's it.  I'd love to hear your thoughts and suggestions.
