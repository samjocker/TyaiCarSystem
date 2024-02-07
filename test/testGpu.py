import tensorflow as tf

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))
