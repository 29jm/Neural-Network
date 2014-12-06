#include "NeuralNetwork.hpp"
#include <vector>
#include <iostream>

using namespace std;

int main() {
	NeuralNetwork nn(1, 2, 1);
	vector<vector<vector<float>>> pat = {
		{{0},{1}},
		{{1},{0}}
	};

	nn.train(pat, 1000);
	nn.test(pat, true);
}
