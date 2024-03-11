from agent_jax import default_states_preprocessor

from jax import jit
import jax.numpy as jnp
import numpy as np
import jax
# states = [np.array([1, 2, 3]), np.array([4,5,6])]
# # # a = default_states_preprocessor(state)
# # # b = default_states_preprocessor(states)
# # print(np.array(states))
# # # print(a)

# a = jnp.array(states)
# print(a. reshape(-1, 1))

# @jit
# def convolve(w, x):  # implementation of 1D convolution/correlation
#     output = []

#     for i in range(1, len(x)-1):
#         output.append(jnp.dot(x[i-1:i+2], w))

#     return jnp.array(output)
# x = np.arange(5)  # signal
# w = np.array([2., 3., 4.])  # window/kernel

# print(convolve(w, x))

#test classes
# arr1 = jnp.array([[[1,2,3]], [[4,5,6]]])
# arr2 = jnp.array([[[3],[2],[1]], [[6],[5],[4]]])
# arr1 = jnp.array([[1,2,3], [4,5,6]])

# arr2 = jnp.array([[[3],[2],[1]], [[6],[5],[4]]])
# arr3 = jnp.array([arr1, arr2])
# print(jax.lax.batch_matmul(jnp.expand_dims(arr1, 1), jnp.expand_dims(arr1, 2)).squeeze())
# arr1 = jnp.array([1,2])
# arr2 = jnp.array([2,1])
# arr3 = jnp.array([[1,2,3], [4,5,6]])
# arr4 = jnp.array([[1,1,1], [2,2,2]])
# arr5 = jnp.array([0,1])
# print(arr3.mean())
# print(jnp.multiply(arr3, 1 - arr5.reshape(-1,1)))
# print(jnp.where((arr1 > arr2).reshape(2,1), arr3, arr4))
key = jax.random.PRNGKey(0)
key, noise_key_ac = jax.random.split(key, 2)
max_action = np.array([1,2,3,200,100])
expl_noise = 0.05
result = jax.random.normal(noise_key_ac, (5,)) * jnp.sqrt(max_action*expl_noise)
print(result.clip(-max_action, max_action))