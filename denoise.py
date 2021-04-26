import numpy as np
import cv2

useRealImage = True

if useRealImage:
    resultFolder = 'results/'
    img = cv2.imread('input/glasses-crop.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(resultFolder + 'input.png', img)
    noisySignal = np.asarray(img)
    noisySignal = noisySignal.astype(np.float32)
    rows = noisySignal.shape[0]
    cols = noisySignal.shape[1]
    totalNumbers = rows * cols
else:
    rows = 4
    cols = 10
    totalNumbers = rows * cols
    noisySignal = np.zeros((rows, cols))
    noisySignal = noisySignal.astype(np.float32)
    for i in range(rows):
        for j in range(cols):
            noisySignal[i][j] = 10
            if i * cols + j == 11 or i * cols + j == 22 or i * cols + j == 35:
                noisySignal[i][j] = 500.0

# the smaller the value, the more confident i am about the constraint
SIGMAMeas = np.dtype(np.float32).type(1)
SIGMASmooth = np.dtype(np.float32).type(0.3)

# dictionaries. i assume that the id of the node is the same as the dictionary location of the node that i want to
# access
variableNodes = {}
factorNodes = {}


class VariableNode:
    def __init__(self, variableID, mu, sigma, priorID, leftID, rightID, upID, downID):
        self.variableID = variableID
        self.mu = mu
        self.sigma = sigma
        self.priorID = priorID
        self.leftID = leftID
        self.rightID = rightID
        self.upID = upID
        self.downID = downID
        self.out_eta = np.zeros(mu.shape)
        self.out_lambda_prime = np.zeros(sigma.shape)

    def getEta(self):
        return self.out_eta

    def getLambdaPrime(self):
        return self.out_lambda_prime

    def getMu(self):
        return self.mu

    def getSigma(self):
        return self.sigma

    def beliefUpdate(self):
        eta_here = np.array([[0.0]]).astype(np.float32)
        lambda_prime_here = np.array([[0.0]]).astype(np.float32)
        factorIDs = [self.priorID, self.leftID, self.rightID, self.upID, self.downID]

        for fID in factorIDs:
            # sometimes i don't have a factor to my left or right and this id is set to -1
            if fID == -1:
                continue
            eta_here += factorNodes[fID].getEta()
            lambda_prime_here += factorNodes[fID].getLambdaPrime()

        if lambda_prime_here == 0.0:
            print('Lambda prime is zero in belief update, something is wrong')
            exit(0)

        self.sigma = np.linalg.inv(lambda_prime_here)
        self.mu = self.sigma * eta_here
        self.out_eta = eta_here
        self.out_lambda_prime = lambda_prime_here

    def computeMessageLeft(self):
        self.beliefUpdate()
        if self.leftID == -1:
            return

        eta_inward = factorNodes[self.leftID].getEta()
        lambda_prime_inward = factorNodes[self.leftID].getLambdaPrime()

        self.out_eta = self.out_eta - eta_inward
        self.out_lambda_prime = self.out_lambda_prime - lambda_prime_inward

    def computeMessageRight(self):
        self.beliefUpdate()
        if self.rightID == -1:
            self.out_eta = 0
            self.out_lambda_prime = 0
            return

        eta_inward = factorNodes[self.rightID].getEta()
        lambda_prime_inward = factorNodes[self.rightID].getLambdaPrime()

        self.out_eta = self.out_eta - eta_inward
        self.out_lambda_prime = self.out_lambda_prime - lambda_prime_inward

    def computeMessageUp(self):
        self.beliefUpdate()
        if self.upID == -1:
            self.out_eta = 0
            self.out_lambda_prime = 0
            return

        eta_inward = factorNodes[self.upID].getEta()
        lambda_prime_inward = factorNodes[self.upID].getLambdaPrime()

        self.out_eta = self.out_eta - eta_inward
        self.out_lambda_prime = self.out_lambda_prime - lambda_prime_inward

    def computeMessageDown(self):
        self.beliefUpdate()
        if self.downID == -1:
            self.out_eta = 0
            self.out_lambda_prime = 0
            return

        eta_inward = factorNodes[self.downID].getEta()
        lambda_prime_inward = factorNodes[self.downID].getLambdaPrime()

        self.out_eta = self.out_eta - eta_inward
        self.out_lambda_prime = self.out_lambda_prime - lambda_prime_inward


# h(x_s) = y, where the state is x_s = [y_s]. dh/dy = 1, so the Jacobian is just [1].
class MeasurementNode:
    def __init__(self, factorID, z, lambdaIn, variableID):
        self.factorID = factorID
        J = np.array([[1.0]]).astype(np.float32)
        self.z = z
        self.lambdaIn = lambdaIn
        # J.T * lambdaIn * [Jx_0 + z_s - h(x_0)] = J.T * lambdaIn * [y_0 + z_s - y_0]
        # These linearisations do not depend on the input so they never change
        self.eta = np.matmul(J.T, lambdaIn) * z
        self.lambdaPrime = np.matmul(np.matmul(J.T, lambdaIn), J)
        self.variableID = variableID
        self.N_sigma = np.sqrt(lambdaIn[0][0]) * 1
        self.variableEta = self.eta
        self.variableLambdaPrime = self.lambdaPrime

    def printFactor(self):
        print('Factor ', self.factorID, ' with measurement ', self.z, ' connected to var ', self.variableID)

    def computeHuberScale(self):
        h = self.z - variableNodes[self.variableID].getMu()[0][0]
        ms = np.sqrt(h * self.lambdaIn[0][0] * h)
        if ms > self.N_sigma:
            k_r = 2.0 * self.N_sigma / ms - (self.N_sigma ** 2) / (ms ** 2)
            return k_r
        return 1

    def computeMessage(self):
        kr = self.computeHuberScale()
        self.variableEta = self.eta * kr
        self.variableLambdaPrime = self.lambdaPrime * kr

    # nothing changes here because this is a unary factor - nothing to multiply or marginalise
    # this is what is should modify to implement the robust factors.
    def getEta(self):
        return self.variableEta

    def getLambdaPrime(self):
        return self.variableLambdaPrime


# h(x_1, x_2) = y_2 - y_1, where the state is x_s = [y_s]. dh/dy_1 = -1, dh/dy_2 = 1 so the Jacobian is just [-1, 1].
class SmoothnessNode:
    def __init__(self, factorID, lambdaIn, prevID, afterID):
        self.factorID = factorID
        J = np.array([[-1, 1]]).astype(np.float32)
        # J transpose * lambda * [ [-1, 1].T * [x1, x2] + 0 - (x2 - x1) ] = J.T * lambda * 0.
        # left is eta[0] and right is eta[1].
        # These never get changed because they do not depend on the input variables so linearisation always
        # has the same form
        self.lambdaIn = lambdaIn
        self.eta = np.matmul(J.T, lambdaIn) * 0
        self.lambda_prime = np.matmul(np.matmul(J.T, lambdaIn), J)
        # these store variable messages
        self.variable_eta = np.array([[0.0]]).astype(np.float32)
        self.variable_lambda = np.array([[0.0]]).astype(np.float32)
        # ids of left and right variable nodes
        self.prevID = prevID
        self.afterID = afterID
        self.N_sigma = np.sqrt(lambdaIn[0][0]) * 1

    def getEta(self):
        return self.variable_eta

    def getLambdaPrime(self):
        return self.variable_lambda

    def computeHuberScale(self):
        h = 0 - (variableNodes[self.afterID].getMu()[0][0] - variableNodes[self.prevID].getMu()[0][0])
        ms = np.sqrt(h * self.lambdaIn[0][0] * h)
        if ms > self.N_sigma:
            k_r = 2.0 * self.N_sigma / ms - (self.N_sigma ** 2) / (ms ** 2)
            return k_r
        return 1

    def computeMessagePrev(self):
        inwardID = self.afterID
        inwardEta = variableNodes[inwardID].getEta()
        inwardLambda = variableNodes[inwardID].getLambdaPrime()

        k_R = self.computeHuberScale()
        eta = np.copy(self.eta)
        lambda_prime = np.copy(self.lambda_prime)

        # left is the first variable in eta and i want to marginalise out the second one (right)
        eta[1] = self.eta[1] + inwardEta
        lambda_prime[1][1] = self.lambda_prime[1][1] + inwardLambda

        eta = eta * k_R
        lambda_prime = lambda_prime * k_R

        eta_a = eta[0]
        eta_b = eta[1]
        lambda_aa = lambda_prime[0][0]
        lambda_ab = lambda_prime[0][1]
        lambda_ba = lambda_prime[1][0]
        lambda_bb = lambda_prime[1][1]

        self.variable_eta = np.array([eta_a - lambda_ab * 1.0 / lambda_bb * eta_b])
        self.variable_lambda = np.array([lambda_aa - lambda_ab * 1.0 / lambda_bb * lambda_ba])

    def computeMessageAfter(self):
        inwardID = self.prevID
        inwardEta = variableNodes[inwardID].getEta()
        inwardLambda = variableNodes[inwardID].getLambdaPrime()

        k_R = self.computeHuberScale()
        eta = np.copy(self.eta)
        lambda_prime = np.copy(self.lambda_prime)

        eta[0] = self.eta[0] + inwardEta
        lambda_prime[0][0] = self.lambda_prime[0][0] + inwardLambda

        eta = eta * k_R
        lambda_prime = lambda_prime * k_R

        eta_a = eta[1]
        eta_b = eta[0]
        lambda_aa = lambda_prime[1][1]
        lambda_ab = lambda_prime[1][0]
        lambda_ba = lambda_prime[0][1]
        lambda_bb = lambda_prime[0][0]

        self.variable_eta = np.array([eta_a - lambda_ab * 1.0 / lambda_bb * eta_b])
        self.variable_lambda = np.array([lambda_aa - lambda_ab * 1.0 / lambda_bb * lambda_ba])

    def computeMessageUp(self):
        # checking that it is a vertical smoothing factor
        if self.afterID - self.prevID > 1:
            # sending a message upwards
            self.computeMessagePrev()

    def computeMessageDown(self):
        # checking that it is a vertical smoothing factor
        if self.afterID - self.prevID > 1:
            # send a message downwards
            self.computeMessageAfter()

    def computeMessageLeft(self):
        # checking that it is a lateral smoothing factor
        if self.afterID - self.prevID == 1:
            # send a message to the left
            self.computeMessagePrev()

    def computeMessageRight(self):
        # checking that it s a lateral smoothing factor
        if self.afterID - self.prevID == 1:
            # send a message to the right
            self.computeMessageAfter()


def printGraph():
    variableIds = variableNodes.keys()
    factorIds = factorNodes.keys()
    print('Printing variables')
    for i in variableIds:
        print(variableNodes[i].variableID, variableNodes[i].priorID, variableNodes[i].upID, variableNodes[i].rightID,
              variableNodes[i].downID, variableNodes[i].leftID)

    print('Printing factors')
    for i in factorIds:
        if hasattr(factorNodes[i], 'variableID'):
            factorNodes[i].printFactor()
        if hasattr(factorNodes[i], 'prevID'):
            print(factorNodes[i].factorID, factorNodes[i].prevID, factorNodes[i].afterID)


# ------- Starting to define the graph here --------
for i in range(rows):
    for j in range(cols):
        varId = i * cols + j
        upId = -1
        downId = -1
        leftId = -1
        rightId = -1
        # the smaller id should be in the primary position
        if i - 1 >= 0:
            up = (i - 1) * cols + j
            upId = (min(up, varId), max(up, varId))
        if i + 1 < rows:
            down = (i + 1) * cols + j
            downId = (min(down, varId), max(down, varId))
        if j - 1 >= 0:
            left = i * cols + j - 1
            leftId = (min(left, varId), max(left, varId))
        if j + 1 < cols:
            right = i * cols + j + 1
            rightId = (min(right, varId), max(right, varId))

        variableNodes[varId] = VariableNode(varId, np.array([[0.0]]).astype(np.float32), np.array([[0.0]]).astype(np.float32), varId, leftId, rightId, upId,
                                            downId)
        factorNodes[varId] = MeasurementNode(varId, noisySignal[i][j], np.array([[1.0 / SIGMAMeas ** 2]]).astype(np.float32), varId)
        if leftId != -1 and leftId not in factorNodes:
            factorNodes[leftId] = SmoothnessNode(leftId, np.array([[1.0 / SIGMASmooth ** 2]]).astype(np.float32), min(leftId), max(leftId))
        if rightId != -1 and rightId not in factorNodes:
            factorNodes[rightId] = SmoothnessNode(rightId, np.array([[1.0 / SIGMASmooth ** 2]]).astype(np.float32), min(rightId),
                                                  max(rightId))
        if upId != -1 and upId not in factorNodes:
            factorNodes[upId] = SmoothnessNode(upId, np.array([[1.0 / SIGMASmooth ** 2]]).astype(np.float32), min(upId), max(upId))
        if downId != -1 and downId not in factorNodes:
            factorNodes[downId] = SmoothnessNode(downId, np.array([[1.0 / SIGMASmooth ** 2]]).astype(np.float32), min(downId), max(downId))

# printGraph()
# ------- Done with defining the graph --------

old_mu = np.array([]).astype(np.float32)
old_sig = np.array([]).astype(np.float32)
iters = 0

np.set_printoptions(precision=8)

while iters < 100:
    print('Iteration ', iters)
    # send a message from the measurement factors
    for key in factorNodes.keys():
        if not isinstance(key, tuple):
            factorNodes[key].computeMessage()

    # every variable is doing a belief update within the message computation
    for key in variableNodes.keys():
        variableNodes[key].computeMessageUp()
    for key in factorNodes.keys():
        if isinstance(key, tuple):
            factorNodes[key].computeMessageUp()

    '''
    for key in factorNodes.keys():
        if not isinstance(key, tuple):
            factorNodes[key].computeMessage()

    for key in variableNodes.keys():
        variableNodes[key].computeMessageRight()
    for key in factorNodes.keys():
        if isinstance(key, tuple):
            factorNodes[key].computeMessageRight()

    for key in factorNodes.keys():
        if not isinstance(key, tuple):
            factorNodes[key].computeMessage()

    for key in variableNodes.keys():
        variableNodes[key].computeMessageDown()
    for key in factorNodes.keys():
        if isinstance(key, tuple):
            factorNodes[key].computeMessageDown()

    for key in factorNodes.keys():
        if not isinstance(key, tuple):
            factorNodes[key].computeMessage()

    for key in variableNodes.keys():
        variableNodes[key].computeMessageLeft()
    for key in factorNodes.keys():
        if isinstance(key, tuple):
            factorNodes[key].computeMessageLeft()

    for key in factorNodes.keys():
        if not isinstance(key, tuple):
            factorNodes[key].computeMessage()

    for key in variableNodes.keys():
        variableNodes[key].beliefUpdate()
    '''

    new_mu = np.array([]).astype(np.float32)
    new_sig = np.array([]).astype(np.float32)
    for key in variableNodes.keys():
        new_mu = np.append(new_mu, variableNodes[key].getMu())
        new_sig = np.append(new_sig, variableNodes[key].getSigma())

    old_mu = np.copy(new_mu)
    old_sig = np.copy(new_sig)
    iters += 1
    if useRealImage:
        result = old_mu.reshape((rows, cols))
        cv2.imwrite(resultFolder + 'result' + str(iters) + '.png', result)

if useRealImage:
    cv2.imshow('result', result / 255.0)
    cv2.waitKey(0)
else:
    np.set_printoptions(precision=8)
    print('signal \n', noisySignal)
    print('mu \n', old_mu.reshape((rows, cols)))
    print(old_mu.dtype)
