The given program executes the given three binary classifier models on the given two datasets to assess their performance.

1. Generative Gaussian Classifier with different standard deviation

2. Generative Gaussian Classifier with same standard deviation

3. Linear SVM classifier

Edit the file path variables in either the notebook file or python file whichever you need to execute.

Execute the given python program by running the command:
python Pr6.py

Alternatively, it is also possible to use the given notebook file by running cells one by one from top to bottom:

Pr6.ipynb

The program automatically executes the following methods on both datasets:

extractXY(file_path) -> returns the input and output numpy arrays x,y by pasring the data of file

genGaussian1(x,y) -> estimate parameters of Gen Gaussian classifier u+,u-,sigma+,sigma-.
                  -> Validate Data on trained model and calculate confusion matrix
                  -> Plot the learned decision boundary

genGaussian2(x,y) -> estimate parameters of Gen Gaussian classifier u+,u-,sigma.
                  -> Validate Data on trained model and calculate confusion matrix
                  -> Plot the learned decision boundary

svmDetails(x,y)   -> Fit the data on given input and output using fit method on a linear SVM classifier from sci-kit learn.
                  -> Validate Data on trained model and calculate confusion matrix
                  -> Plot the learned decision boundary

