<h1> Neural network and decision tree </h1>


Classification of neural network produce output with one softmax layer. I implement and experiment neural network with output as path in a decision tree. Some node of this tree can be "and" or "xor" operator. More complex logic are possible like ("at least two") could be invented but not possible apriori with softmax layer build to choose one decision. 

This tree decision process is coded as multi-softmax output layer. The expected path (ie label) of the neural network is coded with multi-softmax.



My contribution contains :
<ul>
<li> Keras implementation of a new block of layers called "multi-optional-softmax" </li>
<li> A new way to save labels which describe a path on the decision tree. I introduce the storage of arbitrary value -1 to disable backpropagation. </li>
</ul>

Some softmax can be inclusive ("or") or exclusive ("and") in the decision tree. 





<h2> Example with inclusive path </h2>

Are point in the north or south ?
Are point in the west or east ?

Neural network have 2 output softmax :
Label have this pattern : [(P<sub>north</sub>; P<sub>south</sub>) ; (P<sub>west</sub>;P<sub>east</sub>) ]

So  the point : (-0.33;0.44) have label [(0;1);(0;1)] meaning "south-east"
One correct output of neural network can be [(0.2;0.8);(0.1;0.9)]


<h2> Example with exclusive path </h2>

The label contains multi-one-hot-vector. Some have a special value "-1" to disable backpropagation.

<h2> Application </h2>

Those technics can be combined with any loss. We experiment cross entropy and weighted cross entropy.
