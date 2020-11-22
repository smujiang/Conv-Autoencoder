import tensorflow as tf

path = "/infodev1/non-phi-data/junjiang/OvaryCancer_IO/auto_enc_patches_256/OCMC-001/OCMC-001_76928_18816.jpg"

file_contents = tf.io.read_file(path)
image = tf.image.decode_jpeg(file_contents, channels=3)

print(image)

