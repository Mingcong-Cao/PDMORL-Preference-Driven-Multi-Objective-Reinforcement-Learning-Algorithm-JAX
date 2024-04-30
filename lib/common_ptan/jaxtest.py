from agent_jax import default_states_preprocessor

from jax import jit
import jax.numpy as jnp
import numpy as np
import jax
import torch
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
# key = jax.random.PRNGKey(0)
# key, noise_key_ac = jax.random.split(key, 2)
# max_action = np.array([1,2,3,200,100])
# expl_noise = 0.05
# result = jax.random.normal(noise_key_ac, (5,)) * jnp.sqrt(max_action*expl_noise)
# print(result.clip(-max_action, max_action))


# print(output)
key = jax.random.PRNGKey(0)
key, key1, key2 = jax.random.split(key, 3)
x1 = jax.random.normal(key1, (10,12))
x2 = jax.random.normal(key2, (10,12))
x1 = jnp.array([[1,2], [3,4]])
x2 = jnp.array([[-2,1], [4,3]])
dotted = jax.lax.batch_matmul(jnp.expand_dims(x1, 1), jnp.expand_dims(x2, 2)).squeeze()
print("dotted",dotted)
norm1 = jnp.linalg.norm(x1, axis = 1, ord = 2)
norm2 = jnp.linalg.norm(x2, axis = 1,ord = 2)
print("norm",norm1*norm2)
similarity = dotted / jnp.maximum(norm1*norm2, 0.0001)
print(similarity)


def cosine_similarity(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    eps: float = 1e-8,
):
    #compute the cosine similarity as defined in pytorch https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
    #x1, x2 should be 2d arrays
    #return a 1-d array with cosine similarities between two corresponding elements in x1, x2
    dotted = jax.lax.batch_matmul(jnp.expand_dims(x1, 1), jnp.expand_dims(x2, 2)).squeeze()
    # print("dotted",dotted)
    norm1 = jnp.linalg.norm(x1, axis = 1, ord = 2)
    norm2 = jnp.linalg.norm(x2, axis = 1,ord = 2)
    # print("norm",norm1*norm2)
    similarity = dotted / jnp.maximum(norm1*norm2, eps)
    return similarity

#### cosine similariy 
# input1 = np.random.rand(100, 2)*2 - 1
# input2 = np.random.rand(100, 2)*2 - 1
# def sum_logistic(x):
#   return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

# x_small = jnp.arange(3.)
# derivative_fn = grad(sum_logistic)
# print(derivative_fn(x_small))

#grad test of arccos
def clip_arccos(x):
   return jnp.arccos(x.clip(0, 0.9999))[0]
derivative_fn = jax.grad(clip_arccos)

print('jax grad', derivative_fn(jnp.array([0.9999])))

input_torch = torch.Tensor([0.99999]).requires_grad_(True) 
loss = torch.arccos(torch.clamp(input_torch, 0, 0.9999))[0]
loss.backward()
print('torch grad', input_torch.grad)
# re1 = np.array(jnp.arccos(cosine_similarity(jnp.array(input1), jnp.array(input2)).clip(0,0.9999))*180 / jnp.pi)
# cossim = torch.nn.CosineSimilarity(eps=1e-6)
# re2 = torch.rad2deg(torch.acos(torch.clamp(cossim(torch.tensor(input1), torch.tensor(input2)), 0, 0.9999))).numpy()
# mask = (re1 == re2)
# jnp.arccos(cosine_similarity(w_batch_interp, Q1_critic).clip(0,0.9999)) * 180 / jnp.pi
# cos = torch.nn.CosineSimilarity(eps=1e-6)
pass
# angle_term = cosine_similarity(w_batch_interp, Q1_critic).clip(0,0.9999) * 180 / jnp.pi
#np.array(re1)==re2.numpy()