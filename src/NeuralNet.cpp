#include "NeuralNet.hpp"
#include <cmath>


NeuralNet::NeuralNet(const std::vector<unsigned int> p_topology)
{
    for(int i = 0; i < p_topology.size(); i++)
    {
        m_layers.push_back(Layer());

        for(int p = 0; p <= p_topology[i]; p++)
        {
            m_layers.back().push_back(Neuron(p));
        }
    }

    m_layers.back().back().set_value(1.0f);


    for(int i = 0; i < m_layers.size()-1; i++)
    {
        for(int p = 0; p < m_layers[i].size(); p++)
        {
            m_layers[i][p].generate_weights(m_layers[i+1]);
        }
    }


}

NeuralNet::~NeuralNet()
{

}


void NeuralNet::feed_data(const std::vector<float> &p_data)
{
    if(p_data.size() != m_layers[0].size()-1)
        std::cout << "Nuber of inputs incorrect!\n";

    for(int i = 0; i < m_layers[0].size(); i++)
    {
        m_layers[0][i].set_value(p_data[i]);
    }


    for(int i = 1; i < m_layers.size(); i++)
    {
        for(int p = 0; p < m_layers[i].size(); p++)
        {
            m_layers[i][p].calculate_value(m_layers[i-1]);
        }
    }

}

void NeuralNet::back_prop(const std::vector<float> &p_target)
{
    if(p_target.size() != m_layers.back().size()-1)
        std::cout << "Nuber of target values is incorrect!\n";

    for(int i = 0; i < m_layers.back().size()-1; i++)
    {
        m_layers.back()[i].calc_out_gradient(m_layers[m_layers.size()-2], p_target[i]);
    }

    for(int i = m_layers.size()-2; i > 0; i--)
    {
        for(int p = 0; p < m_layers[i].size(); p++)
        {
            m_layers[i][p].calc_hidden_gradient(m_layers[i+1], p_target[i]);
        }
    }

    for(int i = m_layers.size() - 1; i > 0; i--)
    {
        for(int p = 0; p < m_layers[i].size()-1; p++)
        {
            m_layers[i][p].update_weights(m_layers[i-1]);
        }
    }
}

void NeuralNet::get_results(std::vector<float> &p_results)
{
    p_results.clear();
    for(int i = 0; i < m_layers.back().size()-1; i++)
    {
        p_results.push_back(m_layers.back()[i].get_value());
    }
}
