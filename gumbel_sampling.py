#!/usr/bin/env python
# coding: utf-8

# #### This notebook accompagnies Figure 4 of the paper:
# #### "A Review of the Gumbel-max Trick and its Extensions for Discrete Stochasticity in Machine Learning" ("https://arxiv.org/abs/2110.01515"  )
# 
# This notebook can be used to gain insights in the relations between Gumbel-max and Gumbel-softmax samples, generated from unnormalized logits or normalized probabilities.

# In[67]:


# Add some helper functions and libraries
import numpy as np
import scipy

def softmax(x, temperature=1.):
    "Numerically stable implementation of tempered softmax"
    x_max = x.max()
    y = np.exp((x - x_max)/temperature)
    return y / y.sum()

def log_softmax(x):
    "Numerically stable implementation of log-softmax"
    x_max = x.max()
    logsumexp = np.log(np.exp(x - x_max).sum())
    return x - x_max - logsumexp

def one_hot(idx, nr_classes):
    "Converts an index to a one-hot vector of length nr_classes"
    return np.eye(nr_classes)[int(idx)]
    


# ##### Indicate the number of classes and three hyperparameters.

# In[20]:


nr_classes = 4

boltzmann_temp = 1. #boltzmann temperature T
GS_temp = 1. # Gumbel-softmax temperature lambda
beta = 1. # Gumbel noise scale beta

# Generate random unnormalized logits a
a = np.random.normal(size=(nr_classes,))

# Draw Gumbels
gumbels = -np.log(-np.log(np.random.uniform(size=(nr_classes,))))


# In[40]:


# Compute (un)normalized logits via various paths and confirm it results in the same output
log_pi_via_pi = np.log(softmax(a/boltzmann_temp))
log_pi_via_a = log_softmax(a/boltzmann_temp)
np.allclose(
    log_pi_via_pi, 
    log_pi_via_a
)


# In[38]:


# Compute Gumbel-Softmax sample via various paths and confirm it results in the same output
log_pi_via_pi_perturbed = log_pi_via_pi + gumbels
unnormalized_log_perturbed = a/boltzmann_temp + gumbels

np.allclose(
    softmax(log_pi_via_pi_perturbed, GS_temp), 
    softmax(unnormalized_log_perturbed, GS_temp)
)


# In[59]:


# Compute hard sample via various paths and confirm it results in the same output
print(np.allclose( 
    np.argmax(log_pi_via_pi_perturbed), 
    np.argmax(softmax(log_pi_via_pi_perturbed, GS_temp)), 
    )
     )

print(np.allclose(
    np.argmax(unnormalized_log_perturbed), 
    np.argmax(softmax(unnormalized_log_perturbed, GS_temp)) 
    )
)

print(np.allclose(
    np.argmax(log_pi_via_pi_perturbed),
    np.argmax(unnormalized_log_perturbed), 
    )
)

gumbel_max_sample = np.argmax(log_pi_via_pi_perturbed)


# In[60]:


# Compute Gumbel-(soft)max sample via Gumbel noise scaling instead of Boltzmann temperature
print(np.allclose(
    np.argmax(softmax(a + beta*gumbels, GS_temp)),
    np.argmax(a + beta*gumbels),
    )
)

print(np.allclose(
    np.argmax(a + beta*gumbels),
    gumbel_max_sample # Compare to earlier computed argmax output
    )
)


# In[64]:


# Move the GS temperature towards zero and confirm that it results in the same output, as taking the one_hot operation of the argmax outputs
GS_temp = 1e-8
print(np.allclose(
    softmax(a + beta*gumbels, GS_temp),
    softmax(unnormalized_log_perturbed, GS_temp),
    )
)

print(np.allclose(
    softmax(a + beta*gumbels, GS_temp),
    one_hot(gumbel_max_sample, nr_classes)
    )
)


# In[1]:


get_ipython().system('jupyter nbconvert --to python "gumbel_sampling"')

