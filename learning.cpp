#include "learning.h"
#include "cfunction.h"

void SmolBrain::SupervisedLearning::Train(BaseModel* model, TrainingDataset* dataset, int epoch, int batchSize, float learningRate) {
	//_ASSERT(model->InputSize() == dataset->inputs.cols() && model->OutputSize() == dataset->inputs.cols() && dataset->inputs.rows() == dataset->outputs.rows());

	std::vector<SmolBrain::SupervisedLearning::Internal::ErrorData> lastBatchErrors;

	for (int i = 0; i < model->layers.size(); i++) {
		SmolBrain::SupervisedLearning::Internal::ErrorData errorData;
		errorData.weightsErrors = model->layers[i]->weights;
		errorData.weightsErrors.setConstant(0);
		errorData.biasesErrors = model->layers[i]->biases;
		errorData.biasesErrors.setConstant(0);
		lastBatchErrors.push_back(errorData);
	}

	int dataIndex = 0;
	for (int nEpoch = 0; nEpoch < epoch;) {

		std::vector<SmolBrain::SupervisedLearning::Internal::ErrorData> batchErrors;
		float loss = 0;

		for (int i = 0; i < batchSize; i++) {

			Eigen::Matrix<float, -1, 1> currentOutputs = model->Compute(dataset->inputs.row(dataIndex % dataset->inputs.rows()));

			Eigen::Matrix<float, -1, 1> expectedOutputs = dataset->outputs.row(dataIndex % dataset->outputs.rows());

			float cost = CFunction::Cost(currentOutputs, expectedOutputs);
			loss = ((loss * dataIndex) + cost) / (dataIndex + 1);

			std::vector<SmolBrain::SupervisedLearning::Internal::ErrorData> currentError = Sgd(model, currentOutputs, expectedOutputs);

			if (i > 0) {
				for (int j = 0; j < currentError.size(); j++) {
					batchErrors[j].weightsErrors *= i + 1;
					batchErrors[j].weightsErrors += currentError[j].weightsErrors;
					batchErrors[j].weightsErrors /= i + 2;
				}
			}
			else {
				batchErrors = currentError;
			}

			dataIndex++;
		}

		printf("Loss - %f\n", loss);

		for (int i = 0; i < batchErrors.size(); i++) {
			batchErrors[i].weightsErrors += (lastBatchErrors[i].weightsErrors * 0.9);
			batchErrors[i].biasesErrors += (lastBatchErrors[i].biasesErrors * 0.9);
			model->layers[i]->weights -= (batchErrors[i].weightsErrors * learningRate);
			model->layers[i]->biases -= (batchErrors[i].biasesErrors * learningRate);
		}

		lastBatchErrors = batchErrors;

		if (dataIndex >= dataset->inputs.rows()) {
			nEpoch++;
			dataIndex = 0;
		}
	}

}

std::vector<SmolBrain::SupervisedLearning::Internal::ErrorData> SmolBrain::SupervisedLearning::Sgd(BaseModel* model, Eigen::Matrix<float, -1, 1> a, Eigen::Matrix<float, -1, 1> y)
{
	std::vector<SmolBrain::SupervisedLearning::Internal::ErrorData> errors(model->layers.size());

	Eigen::MatrixXf blams = CFunction::DerivativeCost(a, y);

	for (int currentLayer = model->layers.size() - 1; currentLayer >= 0; currentLayer--) {
		BaseLayer* cl = model->layers[currentLayer];
		errors[currentLayer].weightsErrors.resize(cl->weights.rows(), cl->weights.cols());

		Eigen::Matrix<float, -1, -1> li = cl->li.transpose().replicate(cl->Size(), 1);
		Eigen::Matrix<float, -1, -1> z = AFunction::DerivativeActivate(cl->z, cl->fType).replicate(1, li.cols());
		Eigen::Matrix<float, -1, -1> b = blams.replicate(1, li.cols());

		errors[currentLayer].weightsErrors = li.array() * z.array() * b.array();
		errors[currentLayer].biasesErrors = AFunction::DerivativeActivate(cl->z, cl->fType).array() * blams.array();

		blams = (cl->weights.array() * z.array() * b.array()).transpose().rowwise().sum();
	}

	return errors;
}
