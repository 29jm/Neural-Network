#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>
#include <string>
#include <iostream>

class NeuralNetwork {
public:
	NeuralNetwork(const int ni, const int nh, const int no, const bool regression = false);

	std::vector<float> update(const std::vector<float>& inputs);
	float backPropagate(const std::vector<float>& targets, const float lr, const float mf);
	std::vector<std::vector<float>> test(const std::vector<std::vector<std::vector<float>>>& patterns, const bool verbose = false);
	void weights();
	void train(const std::vector<std::vector<std::vector<float>>>& patterns,
		const int iterations = 1000, const float lr = 0.5, const float mf = 0.1, const bool verbose = false);
	
	// Helper math functions
	static std::vector<std::vector<float>> makeMatrix(const int width, const int height, const float fill = 0.0);
	static float sigmoid(const float x);
	static float dsigmoid(const float x);

private:	
	// Number of layers (input, hidden, output)
	unsigned int ni;
	unsigned int nh;
	unsigned int no;

	bool regression;
	
	std::vector<float> ai;
	std::vector<float> ah;
	std::vector<float> ao;

	// Weights
	std::vector<std::vector<float>> wi;
	std::vector<std::vector<float>> wo;

	std::vector<std::vector<float>> ci;
	std::vector<std::vector<float>> co;

	std::mt19937 rng;
};

#endif
