import tensorflow as tf
import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

y = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,0]])
logits = np.array([[12,3,2], [3,10,1], [1,2,5], [4,6.5,1.2], [3,6,1]])

y_pred = sigmoid(logits)
E1 = -y*np.log(y_pred) - (1-y) * np.log(1-y_pred)
print E1

sess = tf.Session()
y = np.array(y).astype(np.float64)
E2 = sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
print E2

if E1.all() == E2.all():
    print "True"
else:
    print "False"