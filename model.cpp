#include "model.h"

#pragma region BaseModel
SmolBrain::BaseModel::BaseModel(int iSize)
{
	this->iSize = iSize;
}

int SmolBrain::BaseModel::InputSize()
{
	return iSize;
}

int SmolBrain::BaseModel::OutputSize()
{
	return layers.back()->Size();
}
#pragma endregion

#pragma region SequentialModel
Eigen::Matrix<float, -1, 1> SmolBrain::SequentialModel::Compute(Eigen::Matrix<float, -1, 1> inputs)
{
	Eigen::Matrix<float, -1, 1> outputs = inputs;

	for (auto& layer : layers) {
		outputs = layer->Compute(outputs);
	}

	return outputs;
}

int SmolBrain::SequentialModel::GetIndexConnectedLayerBackward(int index)
{
	return index - 1;
}
#pragma endregion
