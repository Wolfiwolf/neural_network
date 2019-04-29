#pragma once

#include "Neuron.hpp"

class NeuralNet
{
private:
    std::vector<Layer> m_layers;

public:
    NeuralNet(const std::vector<unsigned int> p_topology);
    ~NeuralNet();

    void feed_data(const std::vector<float>& p_data);

    void back_prop(const std::vector<float>& p_target);

    void get_results(std::vector<float>& p_results);

};
