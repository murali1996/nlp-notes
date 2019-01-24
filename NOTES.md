# Re-usable code snippets

## Loss Optimization in Tensorflow

### For gradient descent to minimize loss in tensorflow, you select
- the variable list to be updated
- the loss function and the optimizer to carry out the update step

### tf.train.<optimizer>.minimize(loss, var_list)
- compute_gradients: This is the first part of minimize(). It returns a list of (gradient, variable) pairs where "gradient" is the gradient for "variable". Note that "gradient" can be a Tensor, an IndexedSlices, or None if there is no gradient for the given variable.
- apply_gradients: This is the second part of minimize(). It returns an Operation that applies gradients
- optimizer.compute_gradients wraps tf.gradients(), as you can see here. It does additional asserts 

### Code
```
1. Simple Update of all Variables
vars = tf.trainable_variables();
gradients = tf.gradients(loss, trainable_params);
update_step = tf.train.RMSPropOptimizer(configure.learning_rate).apply_gradients(zip(gradients, trainable_params), global_step=global_step_val);

2. Update with Clipped gradients
clipped_gradients, global_norm_value = tf.clip_by_global_norm(gradients, configure.max_gradient_norm); 
update_step = tf.train.RMSPropOptimizer(configure.learning_rate).apply_gradients(zip(clipped_gradients, trainable_params), global_step=global_step_val);

3. Update Selected Variables
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")
gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, gen_vars) # G Train step
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, disc_vars) # D Train step

4. Other way of clipping gradients
def _ClipIfNotNone(grad):
    if grad is None:
      return grad
    #grad = tf.clip_by_value(grad, -10, 10, name=None)
    grad = tf.clip_by_norm(grad, 5.0)
    return grad
clipped_gradients = [(_ClipIfNotNone(grad), var) for grad, var in gradients]
```