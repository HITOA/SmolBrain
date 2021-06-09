#pragma once
#include <Eigen/Core>

#include "afunction.h"

namespace SmolBrain {

	class BaseLayer {                                 //Base class for any neural layer structure
	public:
		BaseLayer(int iSize, int oSize);
		virtual void Initialize() = 0;
		virtual Eigen::Matrix<float, -1, 1> Compute(Eigen::Matrix<float, -1, 1> inputs) = 0;
		virtual int Size();
	public:
		Eigen::Matrix<float, -1, -1> weights;
		Eigen::Matrix<float, -1, -1> li;
		Eigen::Matrix<float, -1, -1> z;
		Eigen::Matrix<float, -1, 1> biases;
		AFunction::Type fType;
	protected:
		int iSize, oSize;
	};

	class Dense : public BaseLayer {
	public:
		Dense(AFunction::Type fType, int iSize, int oSize);
		void Initialize();
		Eigen::Matrix<float, -1, 1> Compute(Eigen::Matrix<float, -1, 1> inputs);
	};

}