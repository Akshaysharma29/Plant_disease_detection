# Plant Disease Detection
This project is used for predicting the plant diseases using the image of leaf as input.
This is my first intergration project in which I have integrate my trained neural network(CNN) with webd using flask.

There is one issue that is resolved by 

from keras import backend as K
put this part before loading model.
K.clear_session()
