#include "Neuron.hpp"
#include <random>
#include <cmath>

float Neuron::eta = 0.15f;
float Neuron::alpha = 0.5f;

Neuron::Neuron(unsigned int p_index)
{
    m_index = p_index;

    m_value = (rand() % 100) / 100.0f;

}

Neuron::~Neuron()
{

}

void Neuron::calc_hidden_gradient(Layer &p_next_layer, float p_target)
{
    float sum = 0.0f;

    for(int i = 0; i < p_next_layer.size()-1; i++)
    {
        sum += m_weights[i].weight * p_next_layer[i].m_gradient;
    }

    m_gradient = sum * (1.0f - m_value * m_value);

}

void Neuron::calc_out_gradient(Layer &p_prev_layer, float p_target)
{
    m_gradient = (p_target - m_value) * (1.0f - m_value * m_value);
}


void Neuron::update_weights(Layer &p_prev_layer)
{
    for(int i = 0; i < p_prev_layer.size(); i++)
    {
        Neuron& neuron = p_prev_layer[i];

        float old_delta = neuron.m_weights[m_index].delta_weight;

        float new_delta = eta * neuron.get_value() * m_gradient + alpha * old_delta;

        neuron.m_weights[m_index].delta_weight = new_delta;
        neuron.m_weights[m_index].weight += new_delta;

    }
}



void Neuron::generate_weights(const Layer &p_next_layer)
{
    for(int i = 0; i < p_next_layer.size()-1; i++)
    {

        m_weights.push_back(Connection());
        m_weights.back().weight = (rand() % 100) / 100.0f;


    }
}

void Neuron::calculate_value(const Layer &p_prev_layer)
{
    float sum = 0.0f;

    for(int i = 0; i < p_prev_layer.size(); i++)
    {

        sum += p_prev_layer[i].m_weights[m_index].weight * p_prev_layer[i].get_value();

    }

    m_value = tanh(sum);
}
