#include "NeuralNetwork.hpp"
#include <vector>
#include <iostream>

using namespace std;

int main() {
	NeuralNetwork nn(2, 2, 1);
	vector<vector<vector<float>>> pat = {
		{{0, 0}, {0}},
		{{1, 1}, {0}},
		{{1, 0}, {1}},
		{{0, 1}, {1}},
	};

	for (int i = 0; i < 1000; i++) {
		nn.train(pat, 1);
	}

	nn.test(pat, true);
}
