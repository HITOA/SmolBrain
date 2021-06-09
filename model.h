#pragma once
#include <vector>

#include "layers.h"

namespace SmolBrain {

	class BaseModel {
	public:
		BaseModel(int iSize);
		virtual Eigen::Matrix<float, -1, 1> Compute(Eigen::Matrix<float, -1, 1> inputs) = 0;
		virtual int GetIndexConnectedLayerBackward(int index) = 0;
		int InputSize();
		int OutputSize();
	public:
		std::vector<BaseLayer*> layers;
	protected:
		int iSize;
	};

	class SequentialModel : public BaseModel {
	public:
		SequentialModel(int iSize) : BaseModel(iSize) {};
		template<typename T>
		void AddLayer(int size, AFunction::Type fType) {
			BaseLayer* nLayer;

			if (layers.size() > 0) {
				nLayer = new T(fType, layers.back()->Size(), size);
			}
			else {
				nLayer = new T(fType, iSize, size);
			}

			nLayer->Initialize();

			layers.push_back(nLayer);
		}
		Eigen::Matrix<float, -1, 1> Compute(Eigen::Matrix<float, -1, 1> inputs);
		int GetIndexConnectedLayerBackward(int index);
	};

}