"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost:

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y,outputactivationfunction):
        """Return the error delta from the output layer."""
        return (a-y) * outputactivationfunction.derivative_activation_function(z)


class CrossEntropyCost:

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y,outputactivationfunction):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y) * outputactivationfunction.derivative_activation_function(z) / (a*(1-a))

class LogLossCost:

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if ``a`` has a 1.0
        in the slot, then the expression -y*np.log(a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-a*np.log(a)))

    @staticmethod
    def delta(z, a, y,outputactivationfunction):
        """using http://neuralnetworksanddeeplearning.com/chap3.html
        for crossentropy derivative information wrt weights to see that logless
        function is just the first term in cross entropy. The url gave the 
        derivative derivation by parts so I could just take the relevant
        bit for logloss.
        
        look at https://groups.google.com/forum/#!original/comp.ai.neural-nets/bSq89FWKU8w/BO8foL5OWacJ
        for reference to various forms of crossentropy
        and pairing to particular activation functions 
        """
#        wq = np.array([.99,.01,.99,.99,.5,.01,.9,.1,.9])
#        wq.shape = (9,1)
        
#        return (a-y) * outputactivationfunction.derivative_activation_function(z) / (a*(1-a))
#        return (a-y)*(1-a)  # if activation is logistic sigmoid
#        return wq*(a-y)  # from https://groups.google.com/forum/#!original/comp.ai.neural-nets/bSq89FWKU8w/BO8foL5OWacJ
        return (y-a)  # from https://groups.google.com/forum/#!original/comp.ai.neural-nets/bSq89FWKU8w/BO8foL5OWacJ

class SigmoidActivation:
    @staticmethod
    def apply_activation_function(z):
        return 1.0/(1.0+np.exp(-z))
    @staticmethod    
    def derivative_activation_function(z):
        val = 1.0/(1.0+np.exp(-z))
        return val*(1-val)

class TanhActivation:
    @staticmethod
    def apply_activation_function(z):
        return  np.tanh(z)
    @staticmethod    
    def derivative_activation_function(z):
        return 1-np.power(np.tanh(z),2)

class SoftMaxActivation:
    @staticmethod
    def apply_activation_function(z):
        return np.nan_to_num(np.exp(-z)/sum(np.exp(-z)))
    @staticmethod    
    def derivative_activation_function(z):
        val = np.nan_to_num(np.exp(-z)/sum(np.exp(-z)))
        return val*(1-val)

class LinearActivation:
    @staticmethod
    def apply_activation_function(z):
        return  np.array(z)
    @staticmethod    
    def derivative_activation_function(z):
        return np.ones((len(z),1))
        
class SoftPlusActivation:
    @staticmethod
    def apply_activation_function(z):
        return  np.log(1.0+np.exp(z))
    @staticmethod    
    def derivative_activation_function(z):
        return 1.0/(1.0+np.exp(-z))

class ReLUActivation:
    @staticmethod
    def apply_activation_function(z):
        return  np.max(0.0,z)
    @staticmethod    
    def derivative_activation_function(z):
        if z <= 0:
            val = 0.0
        else:
            val = 1.0
        return val
        

def number_of_nodes(vecOfPairs):
    t = 0
    for pair in vecOfPairs:
        t+= pair[1]
    return t                   
#### Main Network class
class Network():

    def __init__(self, sizes, cost=CrossEntropyCost,
                 mainActivationFunction=SigmoidActivation,
                 mainOutputActivationFunction=SigmoidActivation ):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.initializer_node2nodeMap()
        self.cost=cost
        self.activationfunction = mainActivationFunction
        self.outputactivationfunction = mainOutputActivationFunction

    def initializer_node2nodeMap(self):
        """this is a default to set sefl.dropoutNodeToNodeMap """
        nodeToDropMapAllLayers = self.build_node2drop_map(dropoutProportionHiddenLayers=0.0,
                   dropoutProportionInputLayer=0.0,
                   dropoutProportionOutputLayer=0.0)
        node2nodeMap = self.build_node2node_map(nodeToDropMapAllLayers)
        self.dropoutNodeToNodeMap = node2nodeMap 
                
    def initialize_dropout(self,dropoutProportionHiddenLayers=0.5,
                   dropoutProportionInputLayer=0.0,
                   dropoutProportionOutputLayer=0.0,epochnum=1):
        """called at a START or each epoch"""
        
        nodeToDropMapAllLayers = self.build_node2drop_map(dropoutProportionHiddenLayers=dropoutProportionHiddenLayers,
                   dropoutProportionInputLayer=dropoutProportionInputLayer,
                   dropoutProportionOutputLayer=dropoutProportionOutputLayer)
        #print nodeToDropMapAllLayers
        node2nodeMap = self.build_node2node_map(nodeToDropMapAllLayers)
        #print node2nodeMap        
        self.dropoutNodeToNodeMap = node2nodeMap 
        
        sizes_afteDropout = []                
        for i in range(len(nodeToDropMapAllLayers)):
            sizes_afteDropout.append(number_of_nodes(nodeToDropMapAllLayers[i]))
#        print sizes_afteDropout
 
        self.biases = [np.random.randn(y, 1) for y in sizes_afteDropout[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)  
                    for x, y in zip(sizes_afteDropout[:-1], sizes_afteDropout[1:])]

        self.make_weights_from_MasterWeighs_afterDropout()
        self.make_biases_from_MasterBiases_afterDropout()
        

    def make_weights_from_MasterWeighs_afterDropout(self):
        """function that copies weights from self.masterWeights into self.weights"""
        for i in range(len(self.dropoutNodeToNodeMap)-1):
            node2nodeLayer1 = self.dropoutNodeToNodeMap[i]
            node2nodeLayer2 = self.dropoutNodeToNodeMap[i+1]   
            for pairFrom in node2nodeLayer2:
                for pairTo in node2nodeLayer1:
                    self.weights[i][pairFrom[1]][pairTo[1]] = self.masterWeights[i][pairFrom[0]][pairTo[0]]
 

    def make_biases_from_MasterBiases_afterDropout(self):
        """function that copies appropriate self.masterBiases into self.biases"""
        for i in range(1,len(self.dropoutNodeToNodeMap)):
            node2nodeLayer1 = self.dropoutNodeToNodeMap[i]
            for pair in node2nodeLayer1:
                self.biases[i-1][pair[1]][0] = self.masterBiases[i-1][pair[0]][0]
                    
#            self.biases = [np.array(b,copy=True) for b in self.masterBiases]
#            self.weights = [np.array(w,copy=True) for w in self.masterWeights]
 
    def assign_weights_to_MasterWeighs(self):
        """function that copies weights from self.weights to self.masterWeights """
        for i in range(len(self.dropoutNodeToNodeMap)-1):
            node2nodeLayer1 = self.dropoutNodeToNodeMap[i]
            node2nodeLayer2 = self.dropoutNodeToNodeMap[i+1]   
            for pairFrom in node2nodeLayer2:
                for pairTo in node2nodeLayer1:
                    self.masterWeights[i][pairFrom[0]][pairTo[0]] = self.weights[i][pairFrom[1]][pairTo[1]]
                    self.masterdupWeights[i][pairFrom[0]][pairTo[0]] = self.weights[i][pairFrom[1]][pairTo[1]] 
                  
    def assign_biases_to_MasterBiases(self):
        """function that copies appropriate self.biases to self.masterBiases """
        for i in range(1,len(self.dropoutNodeToNodeMap)):
            node2nodeLayer1 = self.dropoutNodeToNodeMap[i]
            for pair in node2nodeLayer1:
                self.masterBiases[i-1][pair[0]][0] = self.biases[i-1][pair[1]][0]
                self.masterdupBiases[i-1][pair[0]][0] = self.biases[i-1][pair[1]][0]

    def assign_halfweights_to_MasterdupWeighs(self):
        """function that copies weights from self.weights to self.masterWeights """

        self.masterdupWeights = [x for x in self.masterdupWeights] #*(1-dropoutProportion)
                   
    def assign_halfbiases_to_MasterdupBiases(self):
        """function that copies appropriate self.biases to self.masterBiases """

        self.masterdupBiases = [x for x in self.masterdupBiases] #*(1-dropoutProportion)

    def assign_halfweights_to_MasterWeighs(self):
        """function that copies weights from self.weights to self.masterWeights """

        self.masterWeights = [x for x in self.masterWeights] 
                   
    def assign_halfbiases_to_MasterBiases(self):
        """function that copies appropriate self.biases to self.masterBiases """

        self.masterBiases = [x for x in self.masterBiases] 

    def adjust_for_dropout(self):
        """this function is called at the END of EACH EPOCH!!!
        This function assigns the weight found during an epoch iteration
        to the self.masterWeights and self.masterBiases.
        """
        self.assign_biases_to_MasterBiases()
        self.assign_weights_to_MasterWeighs()

            
    def build_node2node_map(self,nodeToDropMapAllLayers):
        """call AFTER build_node2drop_map()
        contains [[(2,0),(3,1)],[(0,0),(2,1),(3,2),(5,3)],...]        
        """
        node2nodeMap = []
        for layer_vec in nodeToDropMapAllLayers:
            formed_pairs = []
            count = 0
            for pair in layer_vec:
                if pair[1] > 0:
                    formed_pairs.append((pair[0],count))
                    count += 1
            node2nodeMap.append(formed_pairs)
        return node2nodeMap
        
        
    def build_node2drop_map(self,dropoutProportionHiddenLayers=0.5,
                   dropoutProportionInputLayer=0.0,
                   dropoutProportionOutputLayer=0.0):
        nodeToDropMap = []
        for i in range(len(self.sizes)):
            if i > 0 and i < len(self.sizes)-1 : #--input layer case
                new_layer = self.form_layer_map(self.sizes[i],dropoutProportion = dropoutProportionHiddenLayers)
            elif i == 0: #---input layer
                new_layer = self.form_layer_map(self.sizes[i],dropoutProportion = dropoutProportionInputLayer)
            elif i == len(self.sizes)-1: #output layer  
                new_layer = self.form_layer_map(self.sizes[i],dropoutProportion = dropoutProportionOutputLayer)                                             
            nodeToDropMap.append(new_layer)
        return nodeToDropMap
               
                       
    def form_layer_map(self,size,dropoutProportion = 0.0):            
        layer_map = []
        sum_ = 0
        layer_nodelist = [ i for i in range(size)]
        dropout_list = random.sample(layer_nodelist,int(size*dropoutProportion))
#        print layer_nodelist
#        print dropout_list
        layer_sel_list = [1 for i in range(size)]
        for i in dropout_list:
            layer_sel_list[i] = 0
#        print layer_sel_list
        
        for i in range(size):
#            random_number = np.random.uniform(0,1)
#            if random_number >= dropoutProportion:
            if layer_sel_list[i] == 1:
                layer_map.append((i,1))
                sum_ += 1
        else:
                layer_map.append((i,0))
                
        #if random selection made all 0, then pick one non-zero at random
        if sum_ < 1:
            layer_map = []
            j = np.random.randint(0,size-1)
            for i in range(size):
                if i != j:
                    layer_map.append((i,0))
                else:
                    layer_map.append((i,1))
        return layer_map 

    def default_weight_initializer(self):
        self.masterBiases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.masterWeights = [np.random.randn(y, x)/np.sqrt(x) 
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.masterdupBiases = self.masterBiases
        self.masterdupWeights = self.masterWeights

    def large_weight_initializer(self):
        self.masterBiases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.masterWeights = [np.random.randn(y, x) 
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.masterdupBiases = self.masterBiases
        self.masterdupWeights = self.masterWeights

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        count = len(self.weights) + 1
        for b, w in zip(self.biases, self.weights):
            count = count - 1
            if count != 1:
                a = (self.activationfunction).apply_activation_function(np.dot(w, a)+b)
            else:
                a = (self.outputactivationfunction).apply_activation_function(np.dot(w, a)+b)
        return a

    def feedforward_wholedupnetwork(self, a):
        """Return the output of the network if ``a`` is input."""
        count = len(self.masterdupWeights) + 1
        for b, w in zip(self.masterdupBiases, self.masterdupWeights):
            count = count - 1
            if count != 1:
                a = (self.activationfunction).apply_activation_function(np.dot(w, a)+b)
            else:
                a = (self.outputactivationfunction).apply_activation_function(np.dot(w, a)+b)
        return a

    def feedforward_wholenetwork(self, a):
        """Return the output of the network if ``a`` is input."""
        count = len(self.masterWeights) + 1
        for b, w in zip(self.masterBiases, self.masterWeights):
            count = count - 1
            if count != 1:
                a = (self.activationfunction).apply_activation_function(np.dot(w, a)+b)
            else:
                a = (self.outputactivationfunction).apply_activation_function(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, numOutputNodes,
            epochmodelfilename,
            lmbda = 0.0, 
            evaluation_data=None, 
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False, 
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            dropoutProportion = 0.5
            
#            if j ==0:
            self.initialize_dropout(dropoutProportionHiddenLayers=dropoutProportion,
                                    dropoutProportionInputLayer=0.0,
                                    dropoutProportionOutputLayer=0.0)
            random.shuffle(training_data)
#            random.shuffle(training_data[0:10416])
#            random.shuffle(training_data[10417:24926])
#            random.shuffle(training_data[24927:39333])
#            random.shuffle(training_data[39334:51443])
#            random.shuffle(training_data[51444:53908])
#            random.shuffle(training_data[53909:66629])
#            random.shuffle(training_data[66630:76850])
#            random.shuffle(training_data[76851:84467])
#            random.shuffle(training_data[84468:])
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0,n, mini_batch_size)]
 
#            mini_batches = [
#                training_data[k:n:mini_batch_size]
#                for k in xrange(0, mini_batch_size)]
#            print len(mini_batches[1])        
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, 
                                       lmbda, len(training_data))
#            print self.weights
#            print self.biases
            print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda,numOutputNodes)
                training_cost.append(cost)
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data,numOutputNodes,convert=True)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {} / {}".format(
                    accuracy, n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda,numOutputNodes, convert=True)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data,numOutputNodes)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data,numOutputNodes), n_data)

                self.assign_halfweights_to_MasterdupWeighs()
                self.assign_halfbiases_to_MasterdupBiases()
                if monitor_evaluation_accuracy:
                    accuracy = self.accuracy_wholedupnetwork(evaluation_data,numOutputNodes)
                    evaluation_accuracy.append(accuracy)
                    print "Accuracy of TOTAL network on evaluation data: {} / {}".format(
                        self.accuracy_wholedupnetwork(evaluation_data,numOutputNodes), n_data)

            self.adjust_for_dropout()

            self.save(epochmodelfilename+str(j))
            eta = eta-0.5/float(epochs)
            print eta

        
            if j == epochs-1:
            
                print "got here"
                self.assign_halfweights_to_MasterWeighs()
                self.assign_halfbiases_to_MasterBiases()
                if monitor_evaluation_accuracy:
                    accuracy = self.accuracy_wholenetwork(evaluation_data,numOutputNodes)
                    evaluation_accuracy.append(accuracy)
                    print "Accuracy of TOTAL network on evaluation data: {} / {}".format(
                        self.accuracy_wholenetwork(evaluation_data,numOutputNodes), n_data)
            
        
        
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy, self.masterWeights, self.masterBiases

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #rand_w = [np.multiply(np.random.randint(0,2,w.shape),.1) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
#        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw 
        self.weights = [(1-float(eta)*(float(lmbda)/float(n)))*w-float(
            eta)/float(len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-float(eta)/float(len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        count = len(nabla_w)+1
        for b, w in zip(self.biases, self.weights):
            count = count - 1
            z = np.dot(w, activation)+b
            zs.append(z)
            if count  != 1:
                activation = self.activationfunction.apply_activation_function(z)
            else:
                activation = self.outputactivationfunction.apply_activation_function(z)
            activations.append(activation) 
        # backward pass


        # needs the delta function below to have a 4th input, the outputactivation
        # (self.cost).delta(zs[-1], activations[-1], y,self.outputactivationfunction)
        # and make change in quadratic cost class
        delta = (self.cost).delta(zs[-1], activations[-1], y,self.outputactivationfunction)
        #print delta
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            spv =self.activationfunction.derivative_activation_function(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, numOutputNodes,convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.  

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if numOutputNodes >1:
            if convert:
                results = [(np.argmax(self.feedforward(x)), np.argmax(y)) 
                           for (x, y) in data]
            else:
                results = [(np.argmax(self.feedforward(x)), y)
                            for (x, y) in data]
            return sum(int(x == y) for (x, y) in results) # confusion matrix count
        else:
                results = [(self.feedforward(x) - y)**2  for (x, y) in data]
                return sum(elem for elem in results)/len(results)  # MSE function
            

    def accuracy_wholenetwork(self, data, numOutputNodes,convert=False):

        
        if numOutputNodes >1:
            if convert:
                results = [(np.argmax(self.feedforward_wholenetwork(x)), np.argmax(y)) 
                           for (x, y) in data]
            else:
                results = [(np.argmax(self.feedforward_wholenetwork(x)), y)
                            for (x, y) in data]
            return sum(int(x == y) for (x, y) in results) # confusion matrix count
        else:
                results = [(self.feedforward_wholenetwork(x) - y)**2  for (x, y) in data]
                return sum(elem for elem in results)/len(results)  # MSE function


    def accuracy_wholedupnetwork(self, data, numOutputNodes,convert=False):

        
        if numOutputNodes >1:
            if convert:
                results = [(np.argmax(self.feedforward_wholedupnetwork(x)), np.argmax(y)) 
                           for (x, y) in data]
            else:
                results = [(np.argmax(self.feedforward_wholedupnetwork(x)), y)
                            for (x, y) in data]
            return sum(int(x == y) for (x, y) in results) # confusion matrix count
        else:
                results = [(self.feedforward_wholedupnetwork(x) - y)**2  for (x, y) in data]
                return sum(elem for elem in results)/len(results)  # MSE function


    def total_cost(self, data, lmbda,numOutputNodes, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y,numOutputNodes)
            cost += self.cost.fn(a, y)/len(data)
        print "without regularization term ",cost
        cost += 0.5*float(lmbda)/float(len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.masterWeights],
                "biases": [b.tolist() for b in self.masterBiases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
    
    def predict(self, IndepVarData,numOutputNodes):
        predictedSolution = []
        fulloutput =[]
        for indepVarRow in IndepVarData:
            if numOutputNodes > 1:
                predictedSolution.append(np.argmax(self.feedforward(indepVarRow)))
                fulloutput.append(np.array(self.feedforward(indepVarRow)))
            else:
                predictedSolution.append(self.feedforward(indepVarRow))
                fulloutput.append(self.feedforward(indepVarRow))
        return predictedSolution,fulloutput           

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j,numOutputNodes):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((numOutputNodes, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
 #   return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

#def sigmoid_vec(z):
    
sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
#    return 1-sigmoid(z)**2

sigmoid_prime_vec = np.vectorize(sigmoid_prime)
