#pragma once
#include <Eigen/Core>
#include <cmath>

namespace SmolBrain {
	namespace CFunction {

		inline float Cost(Eigen::Matrix<float, -1, -1> a, Eigen::Matrix<float, -1, -1> b) {
			Eigen::Matrix<float, -1, -1> r = a - b;
			return r.cwiseAbs2().sum() * 0.5;
		}

		inline Eigen::Matrix<float, -1, -1> DerivativeCost(Eigen::Matrix<float, -1, -1> a, Eigen::Matrix<float, -1, -1> b) {
			return a - b;
		}

	}
}