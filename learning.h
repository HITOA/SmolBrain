#pragma once
#include "model.h"

namespace SmolBrain {
	
	struct TrainingDataset {
		Eigen::Matrix<float, -1, -1> inputs;
		Eigen::Matrix<float, -1, -1> outputs;
	};

	namespace SupervisedLearning {
		void Train(BaseModel* model, TrainingDataset* dataset, int epoch = 5, int batchSize = 1);
		std::vector<Eigen::Matrix<float, -1, -1>> Sgd(BaseModel* model, Eigen::Matrix<float, -1, 1> a, Eigen::Matrix<float, -1, 1> y);
	}

}