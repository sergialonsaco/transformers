import tensorflow as tf

from transformers import TFAutoModel


physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
input_ids = tf.convert_to_tensor([128 * [0]])
model = TFAutoModel.from_pretrained("gpt2")

for _ in range(100):
    model(input_ids)
