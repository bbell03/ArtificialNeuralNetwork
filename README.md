# artificial neural networks
# COMP 131: Artificial Intelligence
# Homework 5: Artificial Neural Networks
# Tufts University

### Author(s): Brandon Bell and Yuchen Yang


### CODE DETAILS

  #### IMPLEMENTATION:
  We expect that our solution has been completely and correctly implemented. The code was written in Python 3.


  #### ARCHITECTURE:

  Our solution makes use of an Artificial Neural Network in order to classify plants according to values corresponding to the length and width of both their petals and sepals.

  According to standard procedure, our Neural Network trains on the dataset ANN -- Iris data.txt using forwards propogation followed by backwards propogation to generalize attributes for each type of plant so as to predict them on input.

  We use various helper functions in addition to a Neural Network class complete
  with its own member functions to implement the solution.

  #### FUNCTION(S) IMPLEMENTED INCLUDE:

  Class Neural Network:
    Attributes:
      Input_Size = 0
      Hidden_Size = 0
      Output_Size = 0

      #Clarify
      w0 = None
      w1 = None
      b0 = None
      b1 = None
      lr = None

    Member Functions:
      train
      predict
      predict_and_calculate_accuracy
      predict_one

       #Returns true if input is Iris-setosa
       def encodersetosa(Class):


       #Returns true if input is Iris-veriscolor
       encoderversicolor(Class):


       #Returns true if input is Iris-virginica
       encodervirginica(Class):
           if Class == "Iris-virginica":
               return 1
           else:
               return 0

       #sigmoid function on input x
       sig(x):
           return (1 / (1 + np.exp(-x)))

       #derivative of sigmoid function on input y
       derisig(y):
           return y * (1 - y)


  #### METHODOLOGY:
    Our program is implemented according to the ANN guidelines
    in the homework specification found at https://canvas.tufts.edu/courses/9172/assignments/63634
