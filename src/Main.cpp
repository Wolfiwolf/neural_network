#include<iostream>
#include <time.h>

#include "NeuralNet.hpp"


int main()
{
    srand(time(NULL));

    NeuralNet net({3,4,1});

    for(int i = 0; i < 1000000; i++)
    {

        float a = rand() % 20;
        float b = rand() % 20;
        float c = rand() % 40;

        float sum = a + b;

        net.feed_data({a,b,sum});
        net.back_prop({1});

        if (c == sum)
        {
            c += a;
        }

        net.feed_data({a,b,c});
        net.back_prop({0});

    }

    std::vector<float> results;

    int a;
    int b;
    int sum;
    printf("[0 - 20]\n");
    printf("num1 : ");
    scanf("%d", &a);
    printf("num2 : ");
    scanf("%d", &b);
    printf("sum : ");
    scanf("%d", &sum);

    net.feed_data({(float)a, (float)b, (float)sum});
    net.get_results(results);
    std::cout << (results[0] > 0.5f ? "CORRECT":"INCORRECT") << "\n";


    return 0;
}
