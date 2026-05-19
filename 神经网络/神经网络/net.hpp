#pragma once

#include <random>
#include <string>
#include <vector>

enum {
    value = 0,
    sum = 1,
    Temp = 2,
};

class TrainingSample {
public:
    std::vector<double> in;
    std::vector<double> out;
};

class NET {
public:
    NET(double learning, std::vector<int> layer_, std::string file, std::string saving = "");
    void save(std::string file);
    void Forward(TrainingSample& samp);
    void Backward(TrainingSample& samp);
    void Learning(int turn);
    std::vector<std::vector<double>> test();
    void INIT();
    double get_loss();
    void init();
    double get_random(double sig);
    double sigmoid(double x);
    std::vector<double> Predict(const std::vector<double>& input);
    int InputSize() const;
    int OutputSize() const;
    int GetSampleCount() const;
    double GetTrainAccuracy();

private:
    std::vector<std::vector<std::vector<double>>> b;  // [layer][node][value/sum/Temp]
    std::vector<std::vector<double>> v;               // [layer][node]
    std::vector<std::vector<std::vector<std::vector<double>>>> w;  // [layer][node][next_layer_node][value/sum]
    std::vector<int> Layer;
    std::vector<TrainingSample> training_data;
    bool training;
    std::mt19937 gen;
    double learning_rate;
};
