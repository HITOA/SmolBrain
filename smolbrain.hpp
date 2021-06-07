#pragma once

#include <vector>
#include <cmath>
#include <stdlib.h>
#include <chrono>

namespace ActivationFunction {
	const float e = std::exp(1.0);

	enum class Type {
		Identity,
		Sigmoid,
		ReLU
	};

	float Identity(float x) {
		return x;
	}

	float IdentityD() {
		return 1;
	}

	float Sigmoid(float x) {
		return 1 / (1 + std::pow(e, -x));
	}

	float SigmoidD(float x) {
		return Sigmoid(x) * (1 - Sigmoid(x));
	}

	float ReLU(float x) {
		return x >= 0 ? x : 0;
	}

	float ReLUD(float x) {
		return x >= 0 ? 1 : 0;
	}

	float Activate(float x, Type fType) {
		switch (fType) {
		case Type::Identity: {
			return Identity(x);
		}
		case Type::Sigmoid: {
			return Sigmoid(x);
		}
		case Type::ReLU: {
			return ReLU(x);
		}
		default: {
			return Identity(x);
		}
		}
	}

	float ActivateD(float x, Type fType) {
		switch (fType) {
		case Type::Identity: {
			return IdentityD();
		}
		case Type::Sigmoid: {
			return SigmoidD(x);
		}
		case Type::ReLU: {
			return ReLUD(x);
		}
		default: {
			return IdentityD();
		}
		}
	}
}

#pragma region Layer

struct Link {
	short oIndex;
	short iIndex;
	float weight;
};

class BaseLayer {
public:
	BaseLayer(short inputSize, short outputSize) {
		this->fType = ActivationFunction::Type::Identity;
		this->inputSize = inputSize;
		this->outputSize = outputSize;
		this->bias.resize(outputSize);
		this->lastOutput.resize(outputSize);
	}

	virtual bool Init() = 0;
	virtual std::vector<float>  Compute(std::vector<float> input) = 0;
	virtual float Activate(float x) = 0;
	virtual float DerivativeActivate(float x) = 0;
	virtual float GetWeightIndex(int outputIndex, int inputIndex) = 0;
	virtual void AdjustWeight(float d, int weightIndex) = 0;

	virtual std::vector<Link> GetLinks(int index) {
		std::vector<Link> currentLinks;

		for (int i = 0; i < links.size(); i++) {
			if (links[i].oIndex == index)
				currentLinks.push_back(links[i]);
		}

		return currentLinks;
	}

	virtual float RandomWeight(float m, float M) {
		int seed = abs(std::chrono::system_clock::now().time_since_epoch().count());
		return m + ((float)((rand() + seed) % RAND_MAX) / ((float)RAND_MAX / (M - m)));
	}

public:
	ActivationFunction::Type fType;
	short inputSize;
	short outputSize;
	std::vector<Link> links;
	std::vector<float> bias;
	std::vector<float> lastOutput; //z

};

class Layer : public BaseLayer {
public:
	Layer(short inputSize, short outputSize) : BaseLayer(inputSize, outputSize) {}

	bool Init() {
		links.resize(outputSize * inputSize);

		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < inputSize; j++) {
				links[i * inputSize + j].oIndex = i;
				links[i * inputSize + j].iIndex = j;
				links[i * inputSize + j].weight = RandomWeight(-1, 1);
			}

			bias[i] = 5;
		}

		return false;
	}

	std::vector<float>  Compute(std::vector<float> input) {
		_ASSERT(input.size() == inputSize);
		std::vector<float> output(outputSize);

		for (int i = 0; i < outputSize; i++) {
			float result = 0;

			for (int j = 0; j < inputSize; j++) {
				_ASSERT(links[i * inputSize + j].oIndex == i);
				_ASSERT(links[i * inputSize + j].iIndex == j);

				result += input[j] * links[i * inputSize + j].weight;
			}

			result += bias[i];

			lastOutput[i] = result;
			output[i] = Activate(result);
		}

		return output;
	}

	float Activate(float x) {
		return ActivationFunction::Activate(x, fType);
	}

	float DerivativeActivate(float x) {
		return ActivationFunction::ActivateD(x, fType);
	}

	float GetWeightIndex(int outputIndex, int inputIndex) {
		return outputIndex * inputSize + inputIndex;
	}
	
	void AdjustWeight(float d, int weightIndex) {
		links[weightIndex].weight -= d;
	}
};

#pragma endregion

#pragma region Network

class Network {
public:
	Network(int inputSize) {
		this->inputSize = inputSize;
	}

	template<typename T>
	void AddLayer(int outputSize, ActivationFunction::Type fType) {
		int currentInputSize = layers.size() > 0 ? layers.back()->outputSize : inputSize;

		BaseLayer* newLayer = new T(currentInputSize, outputSize);
		newLayer->fType = fType;
		newLayer->Init();

		layers.push_back(newLayer);
	}

	std::vector<float> Compute(std::vector<float> input) {
		_ASSERT(input.size() == inputSize);

		std::vector<float> output = input;

		for (auto& layer : layers) {
			output = layer->Compute(output);
		}

		return output;
	}

public:
	std::vector<BaseLayer*> layers;
	int inputSize;
};

#pragma endregion

#pragma region Trainer

struct WeightModifier {
	float d;
	int layer;
	int weightIndex;
};

struct TrainingData {
	std::vector<float> inputs;
	std::vector<float> outputs;
	int inputSize;
	int outputSize;

	TrainingData(std::vector<float> inputs, std::vector<float> outputs, Network network) {
		this->inputs = inputs;
		this->outputs = outputs;
		inputSize = network.inputSize;
		outputSize = network.layers.back()->outputSize;
	}
};

class Trainer {
public:
	Trainer(Network* network, int batchSize, int epoch) {
		this->network = network;
		this->batchSize = batchSize;
		this->epoch = epoch;
	}

	void Train(TrainingData* datas, int numberOfEpoch) {
		int inputSize = datas->inputSize;
		int outputSize = datas->outputSize;

		_ASSERT((datas->inputs.size() % inputSize) == 0);
		_ASSERT((datas->outputs.size() % outputSize) == 0);

		

		for (int currentEpoch = 0; currentEpoch < numberOfEpoch; currentEpoch++) {

			printf("Current Epoch : %i\n", currentEpoch + 1);

			for (int currentBatch = 0; currentBatch < epoch; currentBatch++) {

				printf("-Current Batch : %i\n", currentBatch + 1);

				std::vector<WeightModifier> weightModifiers;

				for (int currentSample = 0; currentSample < batchSize; currentSample++) {
					int index = (currentSample + currentBatch * batchSize + currentEpoch * epoch * batchSize) % (datas->inputs.size() / inputSize);
					//printf("--Current sample : %i\n", index);

					std::vector<float> currentSampleOutput = network->Compute(
						std::vector<float>(datas->inputs.begin() + index * inputSize, 
							datas->inputs.begin() + index * inputSize + inputSize));

					std::vector<float> expectedOutput(
						datas->outputs.begin() + index * outputSize,
						datas->outputs.begin() + index * outputSize + outputSize);

					float loss = Cost(currentSampleOutput, expectedOutput); //calculate loss

					printf("loss = %f\n", loss);

					for (int i = 0; i < outputSize; i++) {
						float da = DerivativeCost(currentSampleOutput[i], expectedOutput[i]);

						std::vector<Link> currentLinks = network->layers.back()->GetLinks(i);

						for (auto& currentLink : currentLinks) {
							std::vector<WeightModifier> currentms = CalculateGradient(currentLink.iIndex, -1, i, da, currentLink.weight);
							weightModifiers.insert(weightModifiers.begin(), currentms.begin(), currentms.end());
							weightModifiers = ComputeTree(weightModifiers);
						}
					}
				}

				ApplyWeightModifier(weightModifiers);

			}

		}
	}

	void ApplyWeightModifier(std::vector<WeightModifier> input) {
		for (auto& m : input) {
			network->layers[network->layers.size() + m.layer]->AdjustWeight(m.d, m.weightIndex);
		}
	}

	std::vector<WeightModifier> CalculateGradient(int index, int layer, int inputIndex, float blam, float weight) {
		BaseLayer* forwardLayer = network->layers[network->layers.size() + layer];
		BaseLayer* currentLayer = network->layers[network->layers.size() - 1 + layer];
		//printf("%i\n", network->layers.size());
		float z = forwardLayer->DerivativeActivate(forwardLayer->lastOutput[inputIndex]);
		float a = currentLayer->Activate(currentLayer->lastOutput[index]);

		float d = a * z * blam;
		//network->layers[network->layers.size() + layer]->AdjustWeight(m, inputIndex, index);

		WeightModifier m;
		m.d = d;
		m.layer = layer;
		m.weightIndex = forwardLayer->GetWeightIndex(inputIndex, index);

		std::vector<WeightModifier> r({ m });

		if (layer + network->layers.size() == 1)
			return r;

		float newBlam = weight * z * blam;

		std::vector<Link> currentLinks = currentLayer->GetLinks(index);

		for (auto& currentLink : currentLinks) {
			std::vector<WeightModifier> gradientChunk = CalculateGradient(currentLink.iIndex, layer - 1, index, newBlam, currentLink.weight);
			r.insert(r.end(), gradientChunk.begin(), gradientChunk.end());
		}

		return ComputeTree(r);
	}

	std::vector<WeightModifier> ComputeTree(std::vector<WeightModifier> input) {
		std::vector<WeightModifier> r;
		std::vector<int> weightIndex;

		r.reserve(input.size());
		weightIndex.reserve(input.size());

		for (auto& m : input) {
			int index = IndexOf(weightIndex, (m.layer * 65535) + m.weightIndex);

			if (index == -1) {
				r.push_back(m);
				weightIndex.push_back((m.layer * 65535) + m.weightIndex);
				continue;
			}

			r[index].d = (r[index].d + m.d) / 2;
		}

		return r;
	}

	int IndexOf(std::vector<int> a, int b) {
		for (int i = 0; i < a.size(); i++) {
			if (a[i] == b)
				return i;
		}

		return -1;
	}

	float Cost(std::vector<float> a, std::vector<float> y) { //MSE
		float r = 0;
		
		for (int i = 0; i < a.size(); i++) {
			r += (a[i] - y[i]) * (a[i] - y[i]);
		}

		return 0.5 * r;
	}

	float DerivativeCost(float a, float y) {
		return a - y;
	}

public:
	Network* network;
	int batchSize;
	int epoch;
};

#pragma endregion
