# Stock price prediction using MLP, RNN, and LSTM
Notes: I'm planning to work on an extension of this project. Consider this a foundation building project.

### Overview and motivation
The task of predicting stock market prices is challenging. Stock prediction is of interest to most investors due to its high volatility. Even now, some investors use a combination of technical and fundamental analysis to help them make better decisions about their equity market investments.  A newly popular method many investors have been using is artificial networks. Therefore, this project aims to find out which ANN model has the best performance.

![Image1](https://github.com/RS201918703/Simple-stock-prediction-with-ANN/blob/main/figs/ActualvsPreds.png)

### Description
This is an academic project for ST311. The updated dataset for this project can be found [here](https://www.kaggle.com/datasets/varpit94/facebook-stock-data). This code was last run on 3rd May 2022 with [this version](https://github.com/RS201918703/ST311-Project/blob/main/FB.csv) of the dataset.

### How to run
- Make sure you have the following packages: [Pytorch](https://pytorch.org/), [Pandas](https://pandas.pydata.org/docs/getting_started/install.html), [NumPy](https://numpy.org/install/), [d2l](http://www.d2l.ai/chapter_installation/index.html), [scikit-learn](https://scikit-learn.org/stable/install.html), [math](https://pypi.org/project/python-math/), [matplotlib](https://matplotlib.org/stable/users/installing/index.html), [Seaborn](https://seaborn.pydata.org/installing.html)
- Run the [code](https://github.com/RS201918703/Simple-stock-prediction-with-ANN/blob/main/Project%20code.ipynb) (All the code was run on CPU but use GPU if you wish)

### Improvements
- Bigger sliding window for longer-term data
- Predictions for more number of days (e.g. one month's worth of predictions instead of one)
- Need to experiment with more ANN structures (e.g. hidden layers, combinations of concepts, types of layers)
- A more complex model including external factors apart from prices alone (could use live news)

### Conclusion
I found that I had control over various variables for each model. But what I decided to adjust was the number of epochs, number of neurons in the hidden layer, learning rates, window size, and dropout probability. It was challenging to make a comparison given the differences in the models. The prediction vs actual prices plots indicated that all our models performed well apart from the second MLP model. But by using RMSE to judge the accuracy, I concluded that the LSTM model outperformed the RNN model while the RNN model outperformed the MLP model. However, I understand how the results may be completely different if I tried the same models on different datasets, or changed up the structures slightly.

### References
1. [Linear regression vs ANN in stock prediction - part 1](https://www.diva-portal.org/smash/get/diva2:1564492/FULLTEXT02.pdf)
2. [Linear regression vs ANN in stock prediction - part 2](https://www.researchgate.net/publication/251368933_Stock_Market_Forecasting_Artificial_Neural_Network_and_Linear_Regression_Comparison_in_An_Emerging_Market)
3. [Deep learning - part 1](https://d2l.ai/)
4. [Deep learning - part 2](https://tanthiamhuat.files.wordpress.com/2018/03/deeplearningwithpython.pdf)
5. [Deep learning - part 3](http://alvarestech.com/temp/deep/Python%20Deep%20Learning%20Exploring%20deep%20learning%20techniques,%20neural%20network%20architectures%20and%20GANs%20with%20PyTorch,%20Keras%20and%20TensorFlow%20by%20Ivan%20Vasilev,%20Daniel%20Slater,%20Gianmario%20Spacagna,%20Peter%20Roelants,%20Va%20(z-lib.org).pdf)
6. [Neural networks](https://www.pdfdrive.com/neural-networks-and-deep-learning-a-textbook-e184020999.html)
7. [Activation functions](https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/)
8. [Activation functions study](https://arxiv.org/pdf/1811.03378.pdf)
9. [Activation functions list](https://prateekvishnu.medium.com/activation-functions-in-neural-networks-bf5c542d5fec)
10. [Saddle point problem](https://proceedings.neurips.cc/paper/2014/file/17e23e50bedc63b4095e3d8204ce063b-Paper.pdf)
11. [Weight initializations](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/)
12. [Inheriting classes](https://realpython.com/python-super/#an-overview-of-pythons-super-function)
13. [Dataloaders in Pytorch](https://www.youtube.com/watch?v=c36lUUr864M&t=12080s&ab_channel=PythonEngineer)
14. [Normalising data - part 1](https://www.journaldev.com/45109/normalize-data-in-python)
15. [Normalising data - part 2](https://towardsdatascience.com/why-data-should-be-normalized-before-training-a-neural-network-c626b7f66c7d)
16. [Measuring accuracy](https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234)
17. [MLP model building](https://medium.com/analytics-vidhya/steps-you-should-follow-to-successfully-train-mlp-40a98c3b5bb3)
18. [Hidden layer structures](https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/)
19. [MLP in stock prediction prices - part 1](https://10mohi6.medium.com/super-easy-python-stock-price-forecasting-using-multilayer-perceptron-machine-learning-4f1d1ef9650)
20. [MLP in stock prediction - part 2](https://www.rsisinternational.org/journals/ijrsi/digital-library/volume-5-issue-7/46-50.pdf)
21. [MLP in short-term predictions ](https://www.researchgate.net/publication/220798177_Short-term_stock_price_prediction_using_MLP_in_moving_simulation_mode)
22. [Sliding window method - part 1](https://ieeexplore.ieee.org/document/6136391)
23. [Sliding window method - part 2](https://www.ripublication.com/ijcir17/ijcirv13n5_46.pdf)
24. [Recurrent neural networks introduction](https://www.youtube.com/watch?v=LHXXI4-IEns&ab_channel=TheA.I.Hacker-MichaelPhi)
25. [RNN models in prediction](https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b)
26. [Pytorch RNN implementation - part 1](https://www.youtube.com/watch?v=0_PgWWmauHk&ab_channel=PythonEngineer)
27. [Pytorch RNN implementation - part 2](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_recurrent_neuralnetwork/)
28. [RNN back propagation](https://www.geeksforgeeks.org/ml-back-propagation-through-time/)
29. [RNNs and the vanishing gradient problem - part 1](https://www.superdatascience.com/blogs/recurrent-neural-networks-rnn-the-vanishing-gradient-problem)
30. [RNN and the vanishing gradient - part 2](https://medium.datadriveninvestor.com/how-do-lstm-networks-solve-the-problem-of-vanishing-gradients-a6784971a577)
31. [RNN and LSTM concepts](https://www.youtube.com/watch?v=WCUNPb-5EYI&ab_channel=BrandonRohrer)
32. [LSTM applications](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)
33. [LSTM concepts - part 1](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
34. [LSTM concepts - part 2](https://blog.mlreview.com/understanding-lstm-and-its-diagrams-37e2f46f1714)
35. [LSTM concepts - part 3](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
36. [LSTM units](https://www.pluralsight.com/guides/introduction-to-lstm-units-in-rnn)
37. [LSTM and GRU concepts](https://www.youtube.com/watch?v=8HyCNIVRbSU&t=585s&ab_channel=TheA.I.Hacker-MichaelPhi)
38. [Pytorch LSTM implementation](https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/)

