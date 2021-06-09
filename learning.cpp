#include "learning.h"
#include "cfunction.h"

void SmolBrain::SupervisedLearning::Train(BaseModel* model, TrainingDataset* dataset, int epoch, int batchSize) {
	_ASSERT(model->InputSize() == dataset->inputs.cols() && model->OutputSize() == dataset->inputs.cols() && dataset->inputs.rows() == dataset->outputs.rows());

	int dataIndex = 0;
	for (int nEpoch = 0; nEpoch < epoch;) {

		std::vector<Eigen::Matrix<float, -1, -1>> batchErrors;

		for (int i = 0; i < batchSize; i++) {

			Eigen::Matrix<float, -1, 1> currentOutputs = model->Compute(dataset->inputs.row(dataIndex % dataset->inputs.rows()));

			Eigen::Matrix<float, -1, 1> expectedOutputs = dataset->outputs.row(dataIndex % dataset->outputs.rows());

			//float loss = CFunction::Cost(currentOutputs, expectedOutputs);

			std::vector<Eigen::Matrix<float, -1, -1>> currentError = Sgd(model, currentOutputs, expectedOutputs);

			if (i > 0) {
				for (int j = 0; j < currentError.size(); j++) {
					batchErrors[j] *= i + 1;
					batchErrors[j] += currentError[j];
					batchErrors[j] /= i + 2;
				}
			}
			else {
				batchErrors = currentError;
			}

			dataIndex++;
		}

		for (int i = 0; i < batchErrors.size(); i++) {
			model->layers[i]->weights -= batchErrors[i];
		}

		if (dataIndex >= dataset->inputs.rows()) {
			nEpoch++;
			dataIndex = 0;
		}
	}

}

std::vector<Eigen::Matrix<float, -1, -1>> SmolBrain::SupervisedLearning::Sgd(BaseModel* model, Eigen::Matrix<float, -1, 1> a, Eigen::Matrix<float, -1, 1> y)
{
	std::vector<Eigen::Matrix<float, -1, -1>> errors(model->layers.size());

	Eigen::MatrixXf blams = CFunction::DerivativeCost(a, y);

	for (int currentLayer = model->layers.size() - 1; currentLayer >= 0; currentLayer--) {
		BaseLayer* cl = model->layers[currentLayer];
		errors[currentLayer].resize(cl->weights.rows(), cl->weights.cols());

		Eigen::Matrix<float, -1, -1> li = cl->li.transpose().replicate(cl->Size(), 1);
		Eigen::Matrix<float, -1, -1> z = AFunction::DerivativeActivate(cl->z, cl->fType).replicate(1, li.cols());
		Eigen::Matrix<float, -1, -1> b = blams.replicate(1, li.cols());

		errors[currentLayer] = li.array() * z.array() * b.array();

		blams = (cl->weights.array() * z.array() * b.array()).transpose().rowwise().sum();
	}

	return errors;
}
