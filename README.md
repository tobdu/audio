## Architecture

#### Loss Function
Since the data is categorical and has a large number of categories, any loss function that uses the distance between the ones and zeros, like `root mean square error` would make no sense. We need a loss function that just looks at the number of hits and misses. `categorical cross-entropy` is such a function.
Another, maybe even more fair option would be to calculate the `area under the ROC` directly after every step, but I haven't seen an out-of-the-box version of that.

## Data Augmentation
With such a small dataset, data synthesis is probably a good strategy. However, because audio data has some specific characteristics it's easy to make mistakes when doing this.
#### White Noise
While adding white-noise is a good strategy to make a model more robust, I'm not convinced if it's useful for this specific task. When dealing with audio recognition from a outdoor environment, like what Amazon's Alexa has to do, I'm sure it's very useful. The purpose of adding white-noise would be to make your model better at finding sound patterns in a noisy signal, but these are all studio recordings where a lot of effort went into reducing back-ground noise. So your model will become good at a task that is not really required.  
That's best case. There are also some mistakes one can make with with-noise: 
- White noise can be an indicator for live recordings. By adding it you are removing that information from your data.
- When adding white noise, you probably want to add it before creating the logarithmic spectrograms, otherwise your white-noise ends up being very dense in the lower frequencies and very sparse in the higher frequencies. Which is a type of noise that, as far as I know, doesn't occur in real life.
#### Compression, Reverb, Distortion
I would stay away from any audio effects that are used when recording music, 
since they can be indicative for genre or era. 
For example: reverb became very popular in the 80ies but is never added to classical music, 
distortion is heavily used in punk, but hardly in jazz, classical music is never compressed 
(which is why it is so annoying to hear on an old car-radio), whilst pop music seems to get more compressed every year...
#### Dropout
Dropout is always a safe way to add randomness. It doesn't fake any input data and can therefor not make any wrong assumption about it.
#### Pitch
Changing pitch can be a very good way of making a model accept songs in all keys. I would guess that there is no pattern between key and genre. 
However, there is very good software out there to change the pitch of a single source, 
I am not sure how easy it is to change the pitch of a combined signal like music without distorting it. 
Note: For western music, when you change the pitch you want to stick to the 12-key scale. 
#### Time
Stretching or compressing time a little bit might be a good way of synthesising data, but I would keep the changes small since tempo can be indicative for genre. 
#### Segmenting
My first choice of data augmentation strategy would be to segmenting the the 30 second audio clips into way smaller ones, say 1 or 2 seconds. 
Our ear can recognise a genre in a couple of seconds. This has several benefits:
* You skip the wrong instructing that the model should look at the 30s clips as a whoile.
* By chopping up the clips the number of samples increases.
* If one uses a sliding window, you also add a time-shift aspect to the model.


In crowd-sourced music tag datasets [2,13], most of the
tags are false(0) for most of the clips, which makes accuracy or mean square error inappropriate as a measure.
Therefore we use the Area Under an ROC (Receiver Operating Characteristic) Curve abbreviated as AUC. This measure has two advantages. It is robust to unbalanced datasets
and it provides a simple statistical summary of the performance in a single value. It is worth noting that a random
guess is expected to score an AUC of 0.5 while a perfect
classification 1.0, i.e., the effective range of AUC spans
between [0.5, 1.0].

https://arxiv.org/pdf/1606.00298.pdf


