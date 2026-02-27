# Overview on signal shape
In this section is explained the shape of the input/output of the network.

<a name="inoutshape"></a>
## Input and output shape from the structured neural model
The structured network can be called in two way:
1. The shape of the inputs not sampled are [total time window size, dim] 
Sampled inputs are reconstructed as soon as the maximum size of the time window is known. 
'dim' represents the size of the input if is not 1 means that the input is a vector.
2. The shape of the sampled inputs are [number of samples = batch, size of time window for a sample, dim]
In the example presented before in the first call the shape for `x` are [1,5,1] for `F` are [1,1,1]
in the second call for `x` are [2,5,1] for `F` are [2,1,1]. In both cases the last dimensions is ignored as the input are scalar.
The output of the structured neural model
The outputs are defined in this way for the different cases:
1. if the shape is [batch, 1, 1] the final two dimensions are collapsed result [batch]
2. if the shape is [batch, window, 1] the last dimension is collapsed result [batch, window]
3. if the shape is [batch, window, dim] the output is equal to [batch, window, dim]
4. if the shape is [batch, 1, dim] the output is equal to [batch, 1, dim]
In the example `x_z_est` has the shape of [1] in the first call and [2] because the the window and the dim were equal to 1.

<a name="elementwiseshape"></a>
## Shape of elementwise Arithmetic, Activation, Trigonometric
The shape and time windows remain unchanged, for the binary operators shape must be equal.
```
input shape = [batch, window, dim] -> output shape = [batch, window, dim]
```

<a name="firshape"></a>
## Shape of Fir input/output
The input must be scalar, the fir compress di time dimension (window) that goes to 1. A vector input is not allowed.
The output dimension of the Fir is moved on the last dimension for create a vector output.
```
input shape = [batch, window, 1] -> output shape = [batch, 1, output dimension of Fir = output_dimension]
```

<a name="linearshape"></a>
## Shape of Linear input/output 
The window remains unchanged and the output dimension is user defined.
```
input shape = [batch, window, dimension] -> output shape = [batch, window, output dimension of Linear = output_dimension]
```

<a name="fuzzyshape"></a>
## Shape of Fuzzy input/output
The function fuzzify the input and creates a vector for output.
The window remains unchanged, input must be scalar. Vector input are not allowed.
```
input shape = [batch, window, 1] -> output shape = [batch, window, number of centers of Fuzzy = len(centers)]
```

<a name="partshape"></a>
## Shape of Part and Select input/output
Part selects a slice of the vector input, the input must be a vector.
Select operation the dimension becomes 1, the input must be a vector.
For both operation if there is a time component it remains unchanged.
```
Part input shape = [batch, window, dimension] -> output shape = [batch, window, selected dimension = [j-i]]
Select input shape = [batch, window, dimension] -> output shape = [batch, window, 1]
```

<a name="timepartshape"></a>
## Shape of TimePart, SimplePart, SampleSelect input/output
The TimePart selects a time window from the signal (works like timewindow `tw([i,j])` but in this the i,j are absolute). 
The SamplePart selects a list of samples from the signal (works like samplewindow `sw([i,j])` but in this the i,j are absolute).
The SampleSelect selects a specific index from the signal (works like zeta operation `z(index)` but in this the index are absolute).
For all the operation the shape remains unchanged.
```
SamplePart input shape = [batch, window, dimension] -> output shape = [batch, selected sample window = [j-i], dimension]
SampleSelect input shape = [batch, window, dimension] -> output shape = [batch, 1, dimension]
TimePart input shape = [batch, window, dimension] -> output shape = [batch, selected time window = [j-i]/sample_time, dimension]
```

<a name="localmodelshape"></a>
## Shape of LocalModel input/output
The local model has two main inputs, activation functions and inputs.
Activation functions have shape of the fuzzy
```
input shape = [batch, window, 1] -> output shape = [batch, window, number of centers of Fuzzy = len(centers)]
```
Inputs go through input function and output function. 
The input shape of the input function can be anything as long as the output shape of the input function have the following dimensions
`[batch, window, 1]` so input functions for example cannot be a Fir with output_dimension different from 1.
The input shape of the output function is `[batch, window, 1]` while the shape of the output of the output functions can be any

<a name="parametersshape"></a>
## Shape of Parameters input/output
Parameter shape are defined as follows `[window = sw or tw/sample_time, dim]` the dimensions can be defined as a tuple and are appended to window
When the time dimension is not defined it is configured to 1

<a name="paramfunshape"></a>
## Shape of Parametric Function input/output
The Parametric functions take inputs and parameters as inputs
Parameter dimensions are the same as defined by the parameters if the dimensions are not defined they will be equal to `[window = 1,dim = 1]`
Dimensions of the inputs inside the parametric function are the same as those managed within the Pytorch framework equal to `[batch, window, dim]`
Output dimensions must follow the same convention `[batch, window, dim]