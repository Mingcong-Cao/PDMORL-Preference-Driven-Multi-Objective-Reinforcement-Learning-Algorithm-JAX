from jax import jit
import jax.numpy as jnp
from flax import linen as nn
import jax
import jax.numpy as jnp  # JAX NumPy
from flax.training import train_state
from networks_jax import Actor, Critic


# actor = Actor(
#     state_size=3,
#     action_size=3,
#     reward_size=5,
#     _layer_N = 3,
#     hidden_size=128,
#     max_action=jnp.array([1.,2.,3.])
# )

# state = jnp.ones((1,3))
# preference = jnp.ones((1,3)) / 3
# key1, key2 = jax.random.split(jax.random.PRNGKey(0))
# # print(actor.tabulate(jax.random.PRNGKey(0), state, preference))
# params = actor.init(key2, state, preference) # Initialization call
# print(jax.tree_util.tree_map(lambda x: x.shape, params)) # Checking output shapes

critic = Critic(
    state_size=3,
    action_size=3,
    reward_size=5,
    _layer_N = 3,
    hidden_size=128,
    max_action=jnp.array([1.,2.,3.])
)

state = jnp.ones((1,3))
preference = jnp.ones((1,3)) / 3
action = jnp.ones((1,3))
key1, key2 = jax.random.split(jax.random.PRNGKey(0))
# print(actor.tabulate(jax.random.PRNGKey(0), state, preference))
params = critic.init(key2, state, preference, action) # Initialization call
print(jax.tree_util.tree_map(lambda x: x.shape, params)) # Checking output shapes