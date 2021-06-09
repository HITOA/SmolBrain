#include "layers.h"

#pragma region BaseLayer
SmolBrain::BaseLayer::BaseLayer(int iSize, int oSize) {
	this->iSize = iSize;
	this->oSize = oSize;
}

int SmolBrain::BaseLayer::Size() {
	return oSize;
}
#pragma endregion

#pragma region Dense
SmolBrain::Dense::Dense(AFunction::Type fType, int iSize, int oSize) : BaseLayer(iSize, oSize)
{
	this->fType = fType;
}

void SmolBrain::Dense::Initialize()
{
	this->weights.resize(oSize, iSize);
	this->weights.setRandom();
}

Eigen::Matrix<float, -1, 1> SmolBrain::Dense::Compute(Eigen::Matrix<float, -1, 1> inputs) {
	Eigen::Matrix<float, -1, 1> outputs = this->weights * inputs;
	li = inputs;
	z = outputs;
	return SmolBrain::AFunction::Activate(outputs, fType);
}
#pragma endregion