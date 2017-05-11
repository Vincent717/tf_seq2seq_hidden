# tf_seq2seq_hidden

It is a pity that the original seq2seq module from tensorflow does not support **getting the hidden states** of model, which sometimes we may want to leverage on. 

Of course you can build a seq2seq model by yourself and do whatever you want, but since I am lazy :) I kind of use a brute method to modify the source code of tensorflow in *tensorflow/tensorflow/python/ops/seq2seq.py* to achieve this.

Basically, I add a new tag argument named **return_hidden_state** (default to be False) to function **model_with_buckets** and **return_input_hidden_state** to **embedding_attention_seq2seq** which are functions that we normally call when using seq2seq. To learn more about how to use seq2seq model in tensorflow, please go to https://www.tensorflow.org/tutorials/seq2seq

## How to use it?
1. Copy the *seq2seq.py* and cover *tensorflow/tensorflow/python/ops/seq2seq.py* in your local address
2. If using *seq2seq_model.py* (a well defined model), try 
```python
def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8, return_hidden_states=True)
    sess.run(tf.global_variables_initializer())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      grad_norms, loss, output, input_states, output_states = model.step(sess, encoder_inputs, decoder_inputs,
                                                                         target_weights, bucket_id, False))
      print('input states are ', input_states)
      print('output states are ', output_states)

self_test()
```
3. If do not use *seq2seq_model.py*, try
```python
## define model
mymodel = lambda encoder_inputs, decoder_inputs : tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
          encoder_inputs,
          decoder_inputs,
          return_input_hidden_states=True)
          
## when training
outputs, losses, input_states, output_states = tf.contrib.legacy_seq2seq.model_with_buckets(
            encoder_inputs, decoder_inputs, targets,
            self.target_weights, buckets,
            mymodel,
            return_hidden_states=True)
```

