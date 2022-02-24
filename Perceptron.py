import numpy as np

class Perceptron:
  def __init__(self):
    self._weights = None
    self._learningRate = 0.01
    self._epoch = 10
    self._numFeatures = None
    self._numClass = None
    self._x = None
    self._y = None

  def fit(self, x, y, lr, epoch):
    self._x = np.array(x)
    self._y = np.array(y)
    self._numFeatures = len(x[0])
    self._numClass = len(y[0])
    self._learningRate = lr
    self._epoch = epoch

    self._weights = np.random.rand(self._numClass, self._numFeatures)

    for ep in range(self._epoch):
      listResult = []

      for i in range(len(self._x)):
        linearOutput = self.linearFunction(self._x[i], self._weights)
        activatedOutput = self.activationFunction(linearOutput)
        softmaxOutput = self.softmaxFunction(activatedOutput)
        listResult.append(softmaxOutput)
        self._weights = self.updateWeights(x[i], activatedOutput, y[i], softmaxOutput)

      print("epoch: ", ep, 
        "\tloss: ", self.crossEntropyLoss(np.asarray(listResult), self._y), 
        "\taccuracy: ", self.accuracy(np.asarray(listResult), y)
      )

  def linearFunction(self, inputFeatures, weights):
    return [np.dot(inputFeatures, weight) for weight in weights]
  
  def activationFunction(self, linearResults):
    return [1 / ( 1 + np.exp(-(linearResult))) for linearResult in linearResults]

  def softmaxFunction(self, activatedResults):
    denominator = sum(self.activationFunction(activatedResults))
    return [self.activationFunction([output])[0] / denominator for output in activatedResults]

  def crossEntropyLoss(self, yPredicted, yActual):
    cummulativeLoss = 0
    for i in range(len(yPredicted)):
      loss = -1 * np.dot(yActual[i], np.log(yPredicted[i]))
      cummulativeLoss = cummulativeLoss + loss

    return loss / len(yPredicted)

  def accuracy(self, yPredicted, yActual):
    ypred = np.argmax(yPredicted, axis = 1)
    yact = np.argmax(yActual, axis = 1)
    correct = (yact == ypred)
    return correct.sum() / correct.size


  def updateWeights(self, x, activated, yactual, ypred):
    temp = np.zeros((self._numClass, self._numFeatures))
    
    for i in range(len(temp)): #output
      for j in range(len(temp[0])): #input
        d = (ypred[i] - yactual[i]) * activated[i] * (1 - activated[i]) * x[j]
        temp[i][j] = self._weights[i][j] - (self._learningRate * d)

    return temp

  def predict(self, x):
    x = np.array(x)
    listResult = []

    for i in range(len(x)):
      linearOutput = self.linearFunction(x[i], self._weights)
      activatedOutput = self.activationFunction(linearOutput)
      softmaxOutput = self.softmaxFunction(activatedOutput)
      listResult.append(softmaxOutput)

    return listResult
