#include "layer.h"

Layer::Layer(int numberOfNeurons, int numberOfInputNodes){
	for (int i = 0; i < numberOfNeurons + 1; ++i)
	{
		Neuron newNeuron(numberOfInputNodes + 1);
		neurons.push_back(newNeuron);
	}
}

Layer::Layer(int numberOfInputNeurons){
	std::vector<float> weights;
	for (int i = 0; i < numberOfInputNeurons; ++i)
	{
		weights.push_back(0.f);
	}

	for (int i = 0; i < numberOfInputNeurons + 1; ++i)
	{
		weights[i] = 1.f;
		Neuron newNeuron(numberOfInputNeurons + 1,weights);
		neurons.push_back(newNeuron);
		weights[i] = 0.f;
	}
}


void Layer::ActivateLayer(std::vector<float>& previousLayer){
	for (int i = 0; i < this->neurons.size(); ++i)
	{
		if (i == 0)
		{
			this->neurons[i].value = 1.f;
			this->neurons[i].preSigValue = 1.f;
		}
		else{
			this->neurons[i].Activate(previousLayer);
		}
	}
}

void Layer::ResetLayer(){
	for (int i = 0; i < neurons.size(); ++i)
	{
		neurons[i].ResetNeuronValue();
	}
}

std::vector<float> Layer::GetLayerValues(){
	std::vector<float> values;
	for (int i = 0; i < this->neurons.size(); ++i)
	{
		values.push_back(neurons[i].value);
	}
	return values;
}