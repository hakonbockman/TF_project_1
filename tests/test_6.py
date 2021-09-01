import tensorflow as tf
assert tf.__version__.startswith('2')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("************ TEST 6 **************************")
tf.debugging.set_log_device_placement(True)

try:
  # Specify an invalid GPU device
  with tf.device('/device:GPU:1'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
except RuntimeError as e:
  print(e)