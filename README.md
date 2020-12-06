# Artificial-neural-network
Artificial Neural Networks (ANN) is a supervised learning system built of a 
large number of simple elements, called neurons or perceptrons. Each neuron 
can make simple decisions, and feeds those decisions to other neurons, organized
in interconnected layers. Together, the neural network can emulate almost any 
function, and answer practically any question, given enough training samples
and computing power. 

<p align="center">
   <img src="image/neuron.jpg" height="170"/>
</p><br/>

A “shallow” neural network has only three layers of neurons:<br/>
1. An input layer that accepts the independent variables or inputs of the model<br/>
2. hidden layer<br/>
3. An output layer that generates predictions<br><br/>
<p align="center">
   <img src="image/network.jpg" height="200"/>
</p>

## Predicting whether a person leave the bank or not using keras 

<b>Data set</b>
The data set contain 10,000 samples of the cutomers containing different parameters.<br/>
RowNumber,CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited.<br/>
Independent Variables: <br/>
 . CreditScore <br/>
 . Geography<br/>
 . Gender <br/>
 . Age <br/>
 . Tenure <br/>
 . Balance<br/>
 . NumOfProducts<br/>
 . HasCrCard<br/>
 . IsActiveMember<br/>
 . EstimatedSalary<br/>
  
 Dependent Variables:<br/>
  . Exited<br/>
  
