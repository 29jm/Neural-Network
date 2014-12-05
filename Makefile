CXXFLAGS = -std=c++14 -g

all: main.o NeuralNetwork.o
	$(CXX) $(CXXFLAGS) main.o NeuralNetwork.o -o test

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

NeuralNetwork.o: NeuralNetwork.hpp NeuralNetwork.cpp
	$(CXX) $(CXXFLAGS) -c NeuralNetwork.cpp -o NeuralNetwork.o
