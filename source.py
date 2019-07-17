import numpy as np
import copy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    x[x < 0] = 0
    return x

def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x



#training dataset generator
int2binary = {}
binary_dim = 8

largest_num = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_num)], dtype=np.uint8).T, axis=1)
for i in range(largest_num):
    int2binary[i] = binary[i]

#input variable
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

synapse_0 = 2*np.random.random((input_dim, hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim, output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim, hidden_dim)) - 1

update_0 = np.zeros_like(synapse_0)
update_1 = np.zeros_like(synapse_1)
update_h = np.zeros_like(synapse_h)

#training sample
for j in range(100000):
    #generate a simple random addition problem (a + b = c)
    a_int = np.random.randint(largest_num/2)
    a = int2binary[a_int]

    b_int = np.random.randint(largest_num / 2)
    b = int2binary[b_int]

    c_int = a_int + b_int
    c = int2binary[c_int]

    d = np.zeros_like(c)

    lost = 0

    layer1_values = list()
    layer2_deltas = list()
    layer1_values.append(np.zeros(hidden_dim))

    for pos in range(binary_dim):
        #generate input and output
        x = np.array([[a[binary_dim - pos - 1], b[binary_dim - pos - 1]]])
        y = np.array([[c[binary_dim - pos - 1]]]).T

        #hidden layer (input + prev_hidden_layer
        layer1 = sigmoid(np.dot(x, synapse_0) + np.dot(layer1_values[-1], synapse_h))

        #output layer
        layer2 = sigmoid(np.dot(layer1, synapse_1))

        layer2_lost = (y - layer2)
        layer2_deltas.append(layer2_lost * sigmoid_derivative(layer2))
        lost += np.abs(layer2_lost[0])

        #decode code estimate so can we print it out
        d[binary_dim - pos - 1] = np.round(layer2[0][0])

        #store hidden layer so we can use it in the next timestep
        layer1_values.append(copy.deepcopy((layer1)))

    future_layer1_delta = np.zeros((hidden_dim))

    for pos in range(binary_dim):

        x = np.array([[a[pos], b[pos]]])
        layer1 = layer1_values[-pos - 1]
        prev_layer1 = layer1_values[-pos - 2]

        layer2_delta = layer2_deltas[-pos - 1]
        layer1_delta = (future_layer1_delta.dot(synapse_h.T)
                        + layer2_delta.dot(synapse_1.T)) * \
                       sigmoid_derivative(layer1)

        #update our weight
        update_1 += np.atleast_2d(layer1).T.dot(layer2_delta)
        update_h += np.atleast_2d(prev_layer1).T.dot(layer1_delta)
        update_0 += x.T.dot(layer1_delta)

        future_layer1_delta = layer1_delta

    synapse_0 += update_0 * alpha
    synapse_h += update_h * alpha
    synapse_1 += update_1 * alpha

    update_1 *= 0
    update_h *= 0
    update_0 *= 0

    if (j % 1000 == 0):

        print("Lost:" + str(lost))
        print("Predict:" + str(d))

        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + "+" + str(b_int) + "=" + str(out))
        print("------------")



