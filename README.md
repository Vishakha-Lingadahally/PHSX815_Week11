# PHSX815_Week11: Neural Network-Wine Quality

One of the first things I did while 'playing around' with the code was check how the accurary varies with the number of epochs. I have attached the behavior as a plot in the repository. I had expected the behavior to be somewhat linear when the number of epochs was less than 100, but that did not seem to be the case. After 1000, the accuracy almost flattens at around 0.5.

The other related and interesting behavior was that of the val loss and train loss with the number of epochs. They decrease with increased number of epochs.

Though a slight increase in accuracy was achieved by alterning the number of epochs, it wasn't significant.

Response to a couple of questions from the notebook:

how was this data obtained? what goes into engineering the features? what does "quality" mean?

1. Are there any correlations among features? is this expected? how can we encode this information to the NN or decouple these features?

Ans: Yes, I would expect some correlation among the features in the data. For example, the pH, citric acid content, acidity would be correlated. One way to examine it would be through scatter plots.

2. What does "quality" mean?

Ans: I would imagine that the most effect way to quantify quality would be by assigning weights to the values of each of the features, and summed. Weights are assigned such that values higher than or lower than the optimum will decrease the sum total. That way we will end up with a scale from 'the worst' to 'the best' quality of wine.

3. Do we have any missing or NaN entries?

Ans: No, in the set of data, we do not have any missing or NaN entries (fortunately).

4. Our network performs ok with a training, validation, and test accuracy all around 50-60%. What could you change about the network or inputs? How do you think that would affect the model's predictions?

Ans: One way of course, would be to increase the number of epochs. I have explored that aspect in this homework.
