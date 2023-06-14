#!/usr/bin/env python
# coding: utf-8

# How does unsqueeze help us to solve certain broadcasting problems?
# 

# Broadcasting Compatibility:
# 
# Broadcasting requires the dimensions of the tensors to be compatible, meaning either the dimensions are equal or one of them has a size of 1.
# When performing elementwise operations, tensors with different shapes need to be aligned to enable broadcasting.
# 
# 
# Dimension Expansion with unsqueeze():
# 
# The unsqueeze() function is used to add a new dimension to a tensor at a specified position.
# It takes an argument specifying the position where the new dimension should be inserted. The position is usually represented by an integer or a tuple of integers.
# Each value in the tuple corresponds to the position along a specific dimension where the new dimension should be inserted.
# 
# Solving Broadcasting Problems:
# 
# If two tensors have different shapes and are not compatible for broadcasting, unsqueeze() can be used to adjust their shapes and introduce new dimensions to make them compatible.
# By unsqueezing a tensor along the appropriate dimensions, its shape is modified to match the shape of the other tensor involved in the operation.
# 

# How can we use indexing to do the same operation as unsqueeze?

# Using Indexing for Dimension Expansion:
# 
# Indexing with None or np.newaxis can be used to insert a new axis or dimension at a specific position in an array.
# It allows us to expand the dimensions of the array and reshape it to match the desired shape.
# 
# 
# Achieving the Same Result as unsqueeze():
# 
# To achieve the same operation as unsqueeze(), we can use indexing with None or np.newaxis at the desired position to introduce a new dimension in the array.
# 

# How do we show the actual contents of the memory used for a tensor?
# 

# Create the tensor using the desired library (e.g., PyTorch, TensorFlow).
# Access the underlying data by using the library-specific method (numpy() in PyTorch, numpy() or eval() in TensorFlow) to convert the tensor to a NumPy array.
# Inspect the NumPy array to see the actual contents of the memory.
# By converting the tensor to a NumPy array and inspecting the array, we can access and view the actual data stored in memory for the tensor.

# When adding a vector of size 3 to a matrix of size 3×3, are the elements of the vector added to each row or each column of the matrix? (Be sure to check your answer by running this code in a notebook.)
# 

# Broadcasting in NumPy:
# 
# NumPy performs element-wise operations using broadcasting, which is a set of rules that allows arrays with different shapes to be combined.
# Broadcasting extends the smaller array to match the shape of the larger array, enabling element-wise operations.
# 
# 
# Broadcasting in Matrix-Vector Addition:
# 
# When adding a vector of size 3 to a matrix of size 3x3, broadcasting will align the vector with each column of the matrix.
# The vector will be added to each corresponding element of the columns.

# Do broadcasting and expand_as result in increased memory use? Why or why not?
# 

# Broadcasting:
# 
# Broadcasting allows arrays with different shapes to be combined in element-wise operations without creating explicit copies of the data.
# It achieves this by virtually extending the smaller array to match the shape of the larger array without actually replicating the data in memory.
# Broadcasting operates on the original arrays and does not allocate additional memory for the operation.
# It provides a memory-efficient way to perform operations on arrays of different shapes.
# 
# 
# expand_as:
# 
# The expand_as method (or similar functionality in other libraries) expands the dimensions of a tensor to match the shape of another tensor.
# It does not create new copies of the data or allocate additional memory.
# Instead, it creates a view of the tensor with expanded dimensions, referring to the same memory as the original tensor.
# The expanded view allows for element-wise operations and is memory-efficient since it avoids storing duplicate data.

# Implement matmul using Einstein summation.
# 

# In[1]:


import numpy as np

def matmul(a, b):
    return np.einsum('ij, jk -> ik', a, b)

A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

C = matmul(A, B)
print(C)


# What does a repeated index letter represent on the lefthand side of einsum?
# 

# In[2]:


import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([2, 3, 4])

C = np.einsum('ij,j->i', A, B)
print(C)


# What are the three rules of Einstein summation notation? Why?

# In[3]:


import numpy as np

# Rule 1: Repeated indices imply summation
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.einsum('i,i->', a, b)
print(c)  # Output: 32 (1*4 + 2*5 + 3*6)

# Rule 2: Free indices represent output indices
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
C = np.einsum('ij,jk->ik', A, B)
print(C)  # Output: [[19 22], [43 50]]

# Rule 3: Indices that appear only on one side are summed over
x = np.array([1, 2, 3])
y = np.einsum('i->', x)
print(y)  # Output: 6 (1 + 2 + 3)


# What are the forward pass and backward pass of a neural network?

# Forward Pass:
# 
# In the forward pass, the input data is passed through the neural network, layer by layer, from the input layer to the output layer.
# Each layer applies a set of weights to the input data, followed by an activation function, to generate an output.
# The output of one layer serves as the input to the next layer until the final output is obtained.
# The forward pass computes the predicted output of the neural network based on the given input.
# 
# 
# Backward Pass (Backpropagation):
# 
# The backward pass, also known as backpropagation, is the process of computing gradients of the loss function with respect to the weights of the neural network.
# It starts from the output layer and works backward to the input layer.
# The gradients are computed using the chain rule of calculus, which calculates how changes in the output of a layer affect the loss function and the weights of that layer.
# The gradients are then used to update the weights of the neural network in order to minimize the loss function during the training process.
# Backpropagation enables the neural network to learn and adjust its weights based on the errors between the predicted output and the true output.

# Why do we need to store some of the activations calculated for intermediate layers in the forward pass?
# 

# Gradient computation: During the backward pass, we need to compute the gradients of the loss function with respect to the parameters (weights and biases) of the neural network. These gradients are calculated using the chain rule of calculus, which involves propagating gradients backward through the network.
# 
# Dependency on previous layers: The gradients at a particular layer depend on the activations of the previous layers. As we propagate the gradients backward, we need the activations of the current layer to calculate the gradients of the previous layers.
# 
# Reusing activations: Storing intermediate layer activations allows us to reuse them in subsequent calculations, avoiding redundant computations. The activations are used in both gradient computations and weight updates during the backward pass. Without storing them, we would need to recalculate the activations for each gradient computation, resulting in unnecessary computation.
# 
# Memory efficiency: Storing activations allows us to save memory by avoiding the need to recompute them. In large neural networks, the intermediate layer activations can consume a significant amount of memory. By storing them, we can reuse the memory space instead of allocating new memory for each computation.
# 
# Numerical stability: Some activation functions, such as the sigmoid or exponential functions, can suffer from numerical stability issues when the input values are very large or very small. Storing intermediate activations helps in maintaining numerical stability during the forward and backward pass, as the values can be reused without being recomputed.
# 
# 

# What is the downside of having activations with a standard deviation too far away from 1?
# 

# Vanishing or exploding gradients: Activations too small can lead to vanishing gradients, making it hard for the network to learn. Activations too large can cause exploding gradients, leading to instability during training.
# Difficulty in optimization: Extreme activation values can make optimization challenging, resulting in slow convergence, suboptimal solutions, or failure to converge.
# Saturation of activation functions: Extreme activations can saturate activation functions, causing gradients close to zero and hindering learning.
# Limited information flow: Small activations can lead to information loss, while large activations can overshadow other inputs, limiting the network's capacity and learning capabilities.

# How can weight initialization help avoid this problem?
# 

# Weight initialization sets the initial values of the weights in a neural network.
# By choosing appropriate initialization methods, such as Xavier or He initialization, the weights are set to suitable scales based on the number of inputs and outputs of each layer.
# Properly initialized weights help control the magnitudes of activations during training.
# Balanced activation magnitudes prevent activations from being too small (vanishing gradients) or too large (exploding gradients).
# When activations stay within a desirable range, the network can learn more effectively and converge faster.
# Weight initialization helps ensure stable and efficient learning by promoting balanced gradients and facilitating convergence.

# In[ ]:




