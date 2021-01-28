This part of the repo houses some playground code that might be useful.

evaluation.ipynb is where I developed the data loader and some very basic evaluation of two trading strategies against each other - buy and hold vs portfolio rebalancing.
The putpose of this was just to develop some tools plus implement the idea of monte carlo analysis of trading results (randomised trials with random start dates, length 
of trading period and (in the case of the portfolio balancing) how often the portfolio is rebalanced.  Any trading strategy should be able to outperform buy and hold in a 
randomised start & length in the test period otherwise it's just cherry picking these to show the best results.

chalvatzis_stackabuse.ipynb is the current active file where I am workign through replicating the Chalvatzis paper using the tutorial posted by Usman Malik on Stackabuse.com.
It loads Chalvatzis' exact dataset (assets & dates) and what I think is his LH LSTM architecture is implemented and the results plotted.



