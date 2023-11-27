# GraphGrad

Welcome to GraphGrad! GraphGrad is a tensor + automatic differentiation library written in C++/CUDA with Python bindings (a la PyTorch).

This is a term project for CSCI-739 (Topics in Intelligent Systems: Machine Learning Systems Implementation) built by **[Jack Oh](https://github.com/jack8558)**, **[Quinn Tucker](https://github.com/qxzcode)**, and **[Dade Wood](https://github.com/daw1882)**.

## Setup
C++17 compiler must be installed to use Graphgrad.

Set up with conda environment
```
conda create --name myenv --file requirement.txt
```
This will install all the packages needed for python interface.

To compile, simply run
```
make
```


## How to use
After running make, user should be able to import graphgrad in python

```
python
>> import graphgrad as gg
>> tensor1 = gg.tensor([1,2])
>> tensor1
```

### Ways to construct tensor
- Construct with data: call gg.tensor function with list
```
tensor = gg.tensor([[1,2],[3,4]])
```
- Construct with random values: pass the dimension as argument and returns tensor with random values
```
tensor = gg.rand([2,3])
```


### Supported tensor operations
- neg
- reciprocal
- relu
- binilarize
- exp
- log
- transpose
- reshape
- add or +
- sub or -
- mul or *
- matmul: currently only supports up to 2D tensors for matmul
- pow
- sum

### Computing gradients
- backward (only for scalar tensor)
```

```

### Example

## Classifier

## Optimization Techniques
- Lazy Evaluation
- Caching transposed matrix for matmul optimization
- CSE(Common subexpression elimination)
- CPU optimization
- GPU optimization

## Unit test
```
pytest
```

## Benchmark with Pytorch
```
python tests/benchmark.py
```
