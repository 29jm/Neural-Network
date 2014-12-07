#include "NeuralNetwork.hpp"

using namespace std;

NeuralNetwork::NeuralNetwork(const int nbi, const int nbh, const int nbo, const bool regression)
	: ni(nbi+1), nh(nbh+1), no(nbo), regression(regression), ai(ni, 1.0), ah(nh, 1.0), ao(no, 1.0), rng(time(nullptr))
{
	wi = NeuralNetwork::makeMatrix(ni, nh);
	wo = NeuralNetwork::makeMatrix(nh, no);

	uniform_real_distribution<float> dist(-1, 1);

	for (int i = 0; i < ni; i++) {
		for (int j = 0; j < nh; j++) {
			wi[i][j] = dist(rng);
		}
	}

	for (int j = 0; j < nh; j++) {
		for (int k = 0; k < no; k++) {
			wo[j][k] = dist(rng);
		}
	}

	ci = makeMatrix(ni, nh);
	co = makeMatrix(nh, no);
}

vector<vector<float>> NeuralNetwork::makeMatrix(const int width, const int height, const float fill) {
	vector<vector<float>> matrix;	
	for (int i = 0; i < width; i++) {
		matrix.push_back(vector<float>(height, fill));
	}

	return matrix;
}

float NeuralNetwork::sigmoid(const float x) {
	return tanh(x);
}

float NeuralNetwork::dsigmoid(const float y) {
	return 1.0 - y*y;
}

vector<float> NeuralNetwork::update(const vector<float>& inputs) {
	if (inputs.size() != ni-1) {
		throw string("wrong number of inputs");
	}

	for (unsigned int i = 0; i < ni-1; i++) {
		ai[i] = inputs[i];
	}

	for (unsigned int j = 0; j < nh-1; j++) {
		float total = 0.0;
		for (int i = 0; i < ni; i++) {
			total += ai[i] * wi[i][j];
		}

		ah[j] = sigmoid(total);
	}

	for (unsigned int k = 0; k < no; k++) {
		float total = 0.0;
		for (unsigned int j = 0; j < nh; j++) {
			total += ah[j] * wo[j][k];
		}

		ao[k] = total;
		if (!regression) {
			ao[k] = sigmoid(total);
		}
	}
		
	return ao;
}

float NeuralNetwork::backPropagate(const vector<float>& targets, const float lr, const float mf) {
	if (targets.size() != no) {
		throw string("wrong number of target values");
	}

	vector<float> output_deltas(no, 1.0);
	for (unsigned int k = 0; k < no; k++) {
		output_deltas[k] = targets[k] - ao[k];
		if (!regression) {
			output_deltas[k] = dsigmoid(ao[k])*output_deltas[k];
		}
	}

	vector<float> hidden_deltas(nh, 0.0);
	for (unsigned int j = 0; j < nh; j++) {
		float error = 0.0;
		for (unsigned int k = 0; k < no; k++) {
			error += output_deltas[k]*wo[j][k];
		}

		hidden_deltas[j] = dsigmoid(ah[j])*error;
	}

	for (unsigned int j = 0; j < nh; j++) {
		for (unsigned int k = 0; k < no; k++) {
			float change = output_deltas[k]*ah[j];
			wo[j][k] = wo[j][k] + lr*change + mf*co[j][k];
			co[j][k] = change;
		}
	}

	for (unsigned int i = 0; i < ni; i++) {
		for (unsigned int j = 0; j < nh; j++) {
			float change = hidden_deltas[j]*ai[i];
			wi[i][j] = wi[i][j] + lr*change + mf*ci[i][j];
			ci[i][j] = change;
		}
	}

	float error = 0.0;
	for (unsigned int k = 0; k < targets.size(); k++) {
		error += 0.5*((targets[k]-ao[k])*(targets[k]-ao[k]));
	}

	return error;
}

vector<vector<float>> NeuralNetwork::test(const vector<vector<vector<float>>>& patterns, const bool verbose) {
	vector<vector<float>> tmp;
	for (const auto& p : patterns) {
		if (verbose) {
			cout << "{ ";
			for (unsigned int i = 0; i < ni-1; i++) {
				cout << p[0][i] << " ";
			}
			cout << "}";
			cout << " -> " << update(p[0])[0] << endl;
		}

		tmp.push_back(update(p[0]));
	}

	return tmp;
}

void NeuralNetwork::train(const vector<vector<vector<float>>>& patterns, const int iterations, const float lr, const float mf, const bool verbose) {
	for (int i = 0; i < iterations; i++) {
		float error = 0;

		for (const auto& p : patterns) {
			update(p[0]);
			float tmp = backPropagate(p[1], lr, mf);
			error += tmp;
		}

		if (i % 100 == 0) {
			cout << "error rate = " << error << endl;
		}
	}
}

