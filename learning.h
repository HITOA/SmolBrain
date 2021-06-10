#pragma once
#include "model.h"

namespace SmolBrain {
	
	struct TrainingDataset {
		Eigen::Matrix<float, -1, -1> inputs;
		Eigen::Matrix<float, -1, -1> outputs;
	};

	namespace SupervisedLearning {
		namespace Internal {
			struct ErrorData {
				Eigen::Matrix<float, -1, -1> weightsErrors;
				Eigen::Matrix<float, -1, 1> biasesErrors;
			};
		}
		void Train(BaseModel* model, TrainingDataset* dataset, int epoch = 5, int batchSize = 1, float learningRate = 0.03);
		std::vector<SmolBrain::SupervisedLearning::Internal::ErrorData> Sgd(BaseModel* model, Eigen::Matrix<float, -1, 1> a, Eigen::Matrix<float, -1, 1> y);
		bool IsCorrect(Eigen::Matrix<float, -1, 1> a, Eigen::Matrix<float, -1, 1> y);
	}

}