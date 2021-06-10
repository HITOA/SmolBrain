#pragma once
#include <Eigen/Core>
#include <cmath>

namespace SmolBrain {
	namespace AFunction {

		const double EulerConstant = std::exp(1.0);

		enum class Type {
			Sigmoid,
			ReLU,
			Tanh
		};

		template <typename Derived>
		inline typename Derived Sigmoid(const Eigen::MatrixBase<Derived>& b) {
			Derived a;
			a.resize(b.rows(), b.cols());

			for (int i = 0; i < b.cols(); i++) {
				for (int j = 0; j < b.rows(); j++) {
					a(j, i) = 1 / (1 + std::pow(EulerConstant, -b(j, i)));
				}
			}

			return a;
		}

		template <typename Derived>
		inline typename Derived ReLU(const Eigen::MatrixBase<Derived>& b) {
			Derived a;
			a.resize(b.rows(), b.cols());

			for (int i = 0; i < b.cols(); i++) {
				for (int j = 0; j < b.rows(); j++) {
					a(j, i) = b(j, i) < 0 ? 0 : b(j, i);
				}
			}

			return a;
		}

		template <typename Derived>
		inline typename Derived Tanh(const Eigen::MatrixBase<Derived>& b) {
			return b.array().tanh().matrix();
		}

		template <typename Derived>
		inline typename Derived Activate(const Eigen::MatrixBase<Derived>& b, Type fType) {
			switch (fType) {
			case Type::Sigmoid:
				return Sigmoid(b);
			case Type::ReLU:
				return ReLU(b);
			case Type::Tanh:
				return Tanh(b);
			default:
				return b;
			}
		}

		#pragma region Derivative

		template <typename Derived>
		inline typename Derived DerivativeSigmoid(const Eigen::MatrixBase<Derived>& b) {

			Eigen::Matrix<float, -1, -1> c;
			c.resize(b.rows(), b.cols());
			c.setConstant(1);

			return (Sigmoid(b).array() * (c - Sigmoid(b)).array()).matrix();
		}

		template <typename Derived>
		inline typename Derived DerivativeReLU(const Eigen::MatrixBase<Derived>& b) {
			Derived a;
			a.resize(b.rows(), b.cols());

			for (int i = 0; i < b.cols(); i++) {
				for (int j = 0; j < b.rows(); j++) {
					a(j, i) = b(j, i) < 0 ? 0 : 1;
				}
			}

			return a;
		}

		template <typename Derived>
		inline typename Derived DerivativeTanh(const Eigen::MatrixBase<Derived>& b) {
			return 1 - (b.array().tanh() * b.array().tanh());
		}

		template <typename Derived>
		inline typename Derived DerivativeActivate(const Eigen::MatrixBase<Derived>& b, Type fType) {
			switch (fType) {
			case Type::Sigmoid:
				return DerivativeSigmoid(b);
			case Type::ReLU:
				return DerivativeReLU(b);
			case Type::Tanh:
				return DerivativeTanh(b);
			default:
				return b;
			}
		}

		#pragma endregion


	}
}