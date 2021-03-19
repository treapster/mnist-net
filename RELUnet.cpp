// This is a simple fully-connected network class, RELU for the internal layer and sigmoid for the last. Internal layers contain biases.
// At the time of writing this it was easier for me not to mess with matrix multiplication and just work with every neuron/output weight manually.

#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <cstdlib>

float activation(float x)
{
    return x > 0.0f ? x : 0.0f;
}
float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}
float sigmoid_deriv(float x)
{
    float a = sigmoid(x);
    return a * (1.0f - a);
}
float activation_deriv(float x)
{
    return x > 0.0f ? 1.0f : 0.0f;
}

class Neuron {
public:
    Neuron() {
        grad = 0.0f;
        value = 0.0f;
    }
    std::vector<float> outputWeight;
    float grad;
    float value;
};


float multiply(std::vector<Neuron>& neurons, int);

class Net {
public:
    Net(const std::vector<int>& config);
    Net(const std::vector<int>& config, std::string& pathToWeights);
    std::vector<float> forward(std::vector<float>& inputValues); //returns last layer values
    std::vector<float> getAnswer(); // returns same values from last layer
	std::vector<float> getLayer(int layerNum);
    float getWeight(int i, int j, int k);
    float backprop(std::vector<float>& answer, float lr); // returns loss
    float getValue(int, int);
	void SaveWeightsToFile(std::string& pathToWeights);
private:
    std::vector<std::vector<Neuron>> neurons;
    int LayersCount;
};
Net::Net(const std::vector<int>& config) {
    LayersCount = config.size();
    int rand_range = 100000;
    float weightsDistributionOffset = 0.0f; // I'm not sure if there are reasons to make it differ from zero, but you can try. Theoretically it can be something like 0.1-0.2 to lower the number of dead RELUs.
    for (int i = 0; i < LayersCount; i++) {
        neurons.push_back(std::vector<Neuron>());
        for (int j = 0; j <= config[i]; j++) // '<=' for bias
        {
            if (!(i == LayersCount - 1 && j == config[i])) //bc there's no bias on the last layer
            {
                neurons[i].push_back(Neuron());
                int k = 0;
                neurons[i].back().grad = 0;
                neurons[i][j].value = 1.0;
                while (k < config[i + 1] && i < LayersCount - 1) // condition for i is bc there's no outputweights on the last layer (and config[i+1] is the upper layer size without bias)
                {
                    neurons[i][j].outputWeight.push_back( ( ((rand()%rand_range + 1) * 2.0f * (1.0f + weightsDistributionOffset)) / (rand_range * 1.0f) - 1.0f ) / (1.0f * config[i])) ;
                    k++;
                }
            }
        }
    }
}
Net::Net(const std::vector<int>& config, std::string& pathToWeights) { // same as default constructor
    std::ifstream file;
	file.open(pathToWeights,std::ios::binary);
    LayersCount = config.size();
    if (!file.good()) {
		std::cout << "Failed to load model\n";
		return;
	}
    float weight;
	int SIZE_OF_WEIGHT= sizeof(weight);
    for (int i = 0; i < LayersCount; i++) {
        neurons.push_back(std::vector<Neuron>());
        for (int j = 0; j <= config[i]; j++) {
            if (!(i == LayersCount - 1 && j == config[i])) //bc there's no bias on the last layer
            {
                neurons[i].push_back(Neuron());
                int k = 0;
                neurons[i].back().grad = 0;
                neurons[i][j].value = 1.0;
                while (i < LayersCount - 1 && k < config[i + 1]) {
                    if (file.read((char*) &weight, SIZE_OF_WEIGHT)) {
                        neurons[i][j].outputWeight.push_back(weight);
                    }
                    else {
                        std::cout << "failed to load weights from " << pathToWeights << "\n";
                        break;
                    }
                    k++;
                }
            }
        }
    }
    file.close();
}
void Net::SaveWeightsToFile(std::string& pathToWeights) {
    std::ofstream file;
	file.open(pathToWeights,std::ios::binary);
    int SIZE_OF_WEIGHT = sizeof (float);
	if (!file.good()) {
		std::cout << "Failed to save\n";
		return;
	}
    for (int i = 0; i < LayersCount - 1; i++) {
        int NeuronsCount = neurons[i].size();
        int UpperLayerSize;
        if (i < LayersCount - 2) //if we're on a deep layer, we don't include upper layer's bias
            UpperLayerSize = neurons[i + 1].size() - 1;
        else
            UpperLayerSize = neurons[i + 1].size(); // otherwise upper layer is the last and doesn't have bias

        for (int j = 0; j < NeuronsCount; j++) {
            for (int k = 0; k < UpperLayerSize; k++) {
                file.write((char*) &neurons[i][j].outputWeight[k], SIZE_OF_WEIGHT);
            }
        }
    }
}

std::vector<float> Net::forward(std::vector<float>& inputValues) {
    if (inputValues.size() != neurons[0].size() - 1) // obvious
        std::cout << "Wrong size of the input vector\n";
    else {
        float a;
        int FirstLayerSize = neurons[0].size() - 1;
        for (int i = 0; i < FirstLayerSize; i++) {
            neurons[0][i].value = inputValues[i];
        }
        neurons[0][FirstLayerSize].value = 1.0;
        for (int i = 1; i < LayersCount; i++) {

            int NeuronCount = neurons[i].size();
            if (i != LayersCount - 1)
                neurons[i][NeuronCount - 1].value = 1.0;
            for (int j = 0; j < NeuronCount; j++) {
                if (i < LayersCount - 1) {// if we're not on the last layer
                    if (j == NeuronCount - 1) //we have bias exception
                        neurons[i][j].value = 1;
                    else {
                        a = multiply(neurons[i - 1], j);
                        neurons[i][j].value = activation(a);
                    }
                }
                else { //and straightforward for the last layer
                    a = multiply(neurons[i - 1], j);
                    neurons[i][j].value = sigmoid(a);
                }
            }
        }
    }
    int n = neurons.back().size();
    std::vector<float> out;
    for (int i = 0; i < n; i++)
        out.push_back(neurons.back()[i].value);
    return out;
}
std::vector<float> Net::getAnswer() {
    int n = neurons.back().size();
    std::vector<float> out;
    for (int i = 0; i < n; i++)
        out.push_back(neurons.back()[i].value);
    return out;
}
float Net::backprop(std::vector<float>& answer, float lr) {
    if (answer.size() != neurons[LayersCount - 1].size()){
        std::cout << "Sizes of answers/output layer don't match\n";
        return 0;
    }
    else {
        float loss_total = 0;
        std::vector<float> loss;
        std::vector<float> loss_grad;
        float partial_grad;
        int NeuronsCount;
        int TopNeuronsCount = answer.size();
        for (int i = 0; i < TopNeuronsCount; i++) {
            loss.push_back(0.5 * (answer[i] - neurons[LayersCount - 1][i].value) * (answer[i] - neurons[LayersCount - 1][i].value));
            loss_grad.push_back(answer[i] - neurons[LayersCount - 1][i].value);
            loss_total += loss[i];
        }
        for (int i = LayersCount - 1; i > 0; i--) {
            int upperLayerSize = 0;
            if (i < LayersCount - 1) {
                upperLayerSize = neurons[i + 1].size();
                NeuronsCount = neurons[i].size() - 1;
            }
            else
                NeuronsCount = neurons[i].size(); // no bias on last layer

            NeuronsCount = neurons[i].size();
            for (int j = 0; j < NeuronsCount; j++) {
                float out = neurons[i][j].value;

                if (i == LayersCount - 1) {
                    float err = loss_grad[j];
                    partial_grad = err * (out * (1.0 - out)); // out = sigmoid(x), and sigmoid'(x) is exactly out * (1.0 - out)
                    neurons[i][j].grad = partial_grad;
                }
                else {
                    partial_grad = activation_deriv(out);
                    float part = 0;
                    for (int l = 0; l < upperLayerSize; l++) {
                        if (i < LayersCount - 2 && l == upperLayerSize - 1) // bc bias of upper layer is not connected to us(and first layer doesn't have bias)
                            break;
                        part += neurons[i][j].outputWeight[l] * neurons[i + 1][l].grad;
                    }
                    neurons[i][j].grad = partial_grad * part;
                }
                int PrevousLayerNeuronsCount = neurons[i - 1].size();
                for (int k = 0; k < PrevousLayerNeuronsCount; k++) {
                    neurons[i - 1][k].outputWeight[j] += neurons[i][j].grad * lr * neurons[i - 1][k].value;
                }
            }
        }
        return loss_total;
    }
}
std::vector<float> Net::getLayer(int layerNum) {
	int size;
	if (layerNum < LayersCount - 1) {
		size = neurons[layerNum].size() - 1;
	}
	else {
		size = neurons.back().size();
	}
	std::vector <float> res(size);
	for (int i = 0; i < size; ++i) {
		res[i] = neurons[layerNum][i].value;
	}
	return res;
}
float Net::getWeight(int i, int j, int k) {
    return neurons[i][j].outputWeight[k];
}

float Net::getValue(int i, int j) {
    return neurons[i][j].value;
}

float multiply(std::vector<Neuron>& neuron, int j) {
    float out = 0;
    int k = neuron.size();
    for (int i = 0; i < k; i++) {
        out += neuron[i].outputWeight[j] * neuron[i].value;
    }
    return out;
}

// Congratulations, this is the end. If you somehow happen to compile it, i recommend using -O3. It will make the thing like 10x faster.