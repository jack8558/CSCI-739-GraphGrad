# GraphGrad

Welcome to GraphGrad! GraphGrad is a tensor + automatic differentiation library written in C++/CUDA with Python bindings (a la PyTorch).

This is a term project for CSCI-739 (Topics in Intelligent Systems: Machine Learning Systems Implementation) built by **[Jack Oh](https://github.com/jack8558)**, **[Quinn Tucker](https://github.com/qxzcode)**, and **[Dade Wood](https://github.com/daw1882)**.

## Setup
C++17 compiler must be installed to use Graphgrad.

Set up with conda environment.
Make sure you have miniconda and python3 installed and updated in your machine.
```
-- Create and activate conda env
conda create --name myenv
conda activate myenv

-- Install python packages
pip3 install -r tests/requirement.txt

-- Install cuda-toolkit
conda config --append channels nvidia
conda install cuda-toolkit -c nvidia
```
This will install all the packages needed for python interface.

To compile, simply run:
```
make
```


## How to use
After running make, user should be able to import graphgrad in python

```
python3
>>> import graphgrad as gg
>>> tensor1 = gg.tensor([1,2])
>>> tensor1
<Tensor: dims=[2], data=[1.000000, 2.000000]>

```

### Ways to construct tensor
- Construct with data: call gg.tensor function with list
```
>>> tensor = gg.tensor([[1,2],[3,4]])
>>> tensor
<Tensor: dims=[2, 2], data=[1.000000, 2.000000, 3.000000, 4.000000]>
```
- Construct with random values: pass the dimension as argument and returns tensor with random values
```
>>> tensor = gg.rand([2,3])
>>> tensor
<Tensor: dims=[2, 3], data=[0.049640, 0.684461, 0.721733, 0.942821, 0.729735, 0.754699]>
```


### Supported tensor operations
- *neg(t)*: Returns a new tensor with the negative of the elements of input tensor.
```
>>> tensor = gg.tensor([1,2])
>>> tensor_neg = tensor.neg()
>>> tensor_neg
<Tensor: dims=[2], data=[-1.000000, -2.000000]>
```
- *reciprocal*: Returns a new tensor with the reciprocal of the elements of input.
```
>>> tensor = gg.tensor([1,2])
>>> tensor_recip = tensor.reciprocal()
>>> tensor_recip
<Tensor: dims=[2], data=[1.000000, 0.500000]>
```
- *relu*: Returns a new tensor with the rectified linear unit function element-wise.
```
>>> tensor = gg.tensor([1,-2])
>>> tensor_relu = tensor.relu()
>>> tensor_relu
<Tensor: dims=[2], data=[1.000000, 0.000000]>
```
- *binilarize*: Returns a new tensor where element becomes 1 if input element was 1, otherwise 0.
```
>>> tensor = gg.tensor([1,-3,-1,100])
>>> tensor_bin = tensor.binilarize()
>>> tensor_bin
<Tensor: dims=[4], data=[1.000000, 0.000000, 0.000000, 1.000000]>
```
- *exp*: Returns a new tensor with the exponential of the elements of the input tensor input.
```
>>> tensor = gg.tensor([1,2,3])
>>> tensor_exp = tensor.exp()
>>> tensor_exp
<Tensor: dims=[3], data=[2.718282, 7.389056, 20.085537]>
```
- *log*: Returns a new tensor with log applied element wise.
```

```
- *transpose(dim0, dim1)*: Returns a new tensor that swap dimension of dim0 and dim1.
- *reshape(dims)*: Returns a new reshaped tensor with given dimension.
- *add or +*: Returns a new tensor where element is added elementwise.
- *sub or -*: Returns a new tensor where element is subtracted elementwise.
- *mul or \**: Returns a new tensor where element is multiplied elementwise.
- *matmul*: Performs matrix multiplication. Currently only supports up to 2D tensors for matmul.
- *pow*: Takes the power of each element in t1 with t2 and returns a tensor with the result.
- *sum*: Returns the sum of all elements in the input tensor.

### Computing gradients
- backward (only for scalar tensor): Computes the gradient of current tensor wrt graph leaves. The graph is differentiated using the chain rule.
```

```

### Example

## Classifier

## Optimization Techniques
Below are optimization techniques GraphGrad uses

- Lazy Evaluation: GraphGrad uses lazy evaluation where the values not gets computes until evaluation functions get called.
- Caching transposed matrix for matmul optimization: 
- CSE(Common subexpression elimination): GraphGrad uses CSE to 
- CPU optimization: Multithreading with CPU
- GPU optimization

## Benchmark with Pytorch
```
python tests/benchmark.py
```
