# SmolBrain
Little machine learning/deep learning library for learning purpose.
I use Eigen->blas for the linear algebra calculus with matricies. so you will need it to use it.

Example code : 
```cpp
	SmolBrain::SequentialModel model(2);

	model.AddLayer<SmolBrain::Dense>(10, SmolBrain::AFunction::Type::Sigmoid);
	model.AddLayer<SmolBrain::Dense>(25, SmolBrain::AFunction::Type::Sigmoid);
	model.AddLayer<SmolBrain::Dense>(2, SmolBrain::AFunction::Type::Sigmoid);

	SmolBrain::TrainingDataset dataset;

	dataset.inputs.resize(5, 2);
	dataset.outputs.resize(5, 2);

	dataset.inputs << 1, 1,
		2, 2,
		3, 3,
		4, 4,
		5, 5;

	dataset.outputs << 0, 0,
		0, 0,
		0, 0,
		0, 0,
		0, 0;

	SmolBrain::SupervisedLearning::Train(&model, &dataset, 100, 50, 0.03); //Model, Dataset, Epoch, BatchSize, learningRate
```
