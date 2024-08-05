

We should test the implementations and models also for stocks that dont have a continous up/downtrend.


LSTM with attention blocks vs vanilla LSTM.
LSTSM trained on vairable input lenght sequences. -> Goal is to get better generalisable models.
Train the model on all 4 dimensions on the input, but only do predictions etc on the Close price.
Transformer vs LSTM with attention.

Adding a couple of fourier transforms on the training data. Maybe this can inhernetly encode some of the periodicity in the data?



Results:
Seems that by incresing the numbers of LSTM blocks the model just ends up predicting the average each time. Vanishing gradients?

One step with quasi multistep
0.04089337959057901 RMSE5 on the testset

Multistep in one shot:
0.02738187177941771

I.e a rather good increase compared to the one step with quasi multistep



TODO:
[] - Maybe alter the way we create the sequences. Is there any reason for having the roll for the train sequences and not for the test and validation sequences??
[] - Add the quasi multistep as a part of the model class perhaps. Simplifies a lot of stuff for the training and testing of the models.