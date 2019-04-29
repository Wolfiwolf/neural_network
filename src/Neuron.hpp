#pragma once

#include <iostream>

#include <vector>

class Neuron;

typedef std::vector<Neuron> Layer;

struct Connection
{
    float weight, delta_weight;

};

class Neuron
{
private:
    float m_value;
    std::vector<Connection> m_weights;

    unsigned int m_index;
    float m_gradient;

    static float eta;
    static float alpha;

public:
    Neuron(unsigned int p_index);
    ~Neuron();

    void calc_out_gradient(Layer& p_prev_layer, float p_target);
    void calc_hidden_gradient(Layer& p_next_layer, float p_target);

    void update_weights(Layer& p_prev_layer);

    void generate_weights(const Layer& p_prev_layer);

    void calculate_value(const Layer& p_prev_layer);


    inline void set_value(float p_val)
    {
        m_value = p_val;
    }

    inline float get_value() const
    {
        return m_value;
    }

    inline float get_weight(unsigned int p_index) const
    {
        return m_weights[p_index].weight;
    }


};
