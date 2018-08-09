# how to run?

Open your terminal on Mac:

   cd Desktop

   mkdir MNIST

   cd MNIST

   git clone https://github.com/BattleBird/mnist

   code mnist

   virtualenv .

   pip install tensorflow==1.4.0
   

*** [pylint] E0401:Unable to import 'tensorflow.examples.tutorials.mnist'

Solution:open your VScode's settings, and then add  "python.pythonPath": "/Users/hangxu/Desktop/MNIST/bin/python" to your custom settings.


python download.py

*** cannot download train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz to          MNIST_data/

Solution: use Chrome(because other browers will uncompress these dataset) to open http://yann.lecun.com/exdb/mnist/ and download 4       dataset, then put them into MNIST_data/


   python download.py (again)

   python label.py

   python save.pic.py
   

*** [pylint] E0401:Unable to import 'scipy.misc'

Solution: pip install scipy


   python save.pic.py (again)

*** 'module' object has no attribute 'toimage'

Solution: pip install pillow


   python save.pic.py (again)
   

*** RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88

Reason: the versions of numpy and scipy don't match

Solution: sudo pip uninstall numpy
          sudo pip install numpy==1.14.5
          
          
   python save.pic.py (again) That's OK, there is no warning.

   python softmax_regression.py (accuracy:0.9078)

   python convolutional.py (accuracy:)
