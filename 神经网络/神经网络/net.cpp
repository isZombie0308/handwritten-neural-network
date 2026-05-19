#include "net.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>

using namespace std;

NET::NET(double learning, vector<int> layer_, string file, string saving) {
    mt19937 tem((unsigned)time(nullptr));
    gen = tem;
    Layer = layer_;
    training = saving.empty();
    learning_rate = learning;

    ifstream fin(file, ios::in);
    if (!fin.is_open()) {
        cout << "无法打开文件: " << file << "，将使用空数据集继续初始化。\n";
    } else if (training) {
        string line;
        while (getline(fin, line)) {
            if (line.empty()) {
                continue;
            }
            istringstream iss(line);
            TrainingSample temp;
            double t = 0.0;
            bool ok = true;
            for (int i = 0; i < Layer[0]; i++) {
                if (!(iss >> t)) {
                    ok = false;
                    break;
                }
                temp.in.push_back(t);
            }
            if (!ok) {
                continue;
            }
            for (int i = 0; i < Layer[(int)Layer.size() - 1]; i++) {
                if (!(iss >> t)) {
                    ok = false;
                    break;
                }
                temp.out.push_back(t);
            }
            if (!ok) {
                continue;
            }
            training_data.push_back(temp);
        }
        fin.close();
    } else {
        // Predict mode: no need to preload training samples.
        fin.close();
    }

    if (training && !training_data.empty()) {
        for (auto& s : training_data) {
            if ((int)s.out.size() != Layer.back()) {
                continue;
            }
            for (int k = 0; k < (int)s.out.size(); k++) {
                s.out[k] = (s.out[k] > 0.5) ? 0.9 : 0.1;
            }
        }
    }

    bool weights_loaded = false;

    for (int i = 0; i < (int)Layer.size(); i++) {
        b.push_back(vector<vector<double>>(Layer[i], vector<double>(3, 0.0)));
        v.push_back(vector<double>(Layer[i], 0.0));
        if (i < (int)Layer.size() - 1) {
            w.push_back(vector<vector<vector<double>>>(
                Layer[i], vector<vector<double>>(Layer[i + 1], vector<double>(2, 0.0))));
        }
    }

    if (!training) {
        ifstream load_fin(saving, ios::in);
        if (load_fin.is_open()) {
            for (int i = 0; i < (int)Layer.size(); i++) {
                if (i == 0) {
                    for (int j = 0; j < Layer[i]; j++) {
                        for (int k = 0; k < Layer[i + 1]; k++) {
                            load_fin >> w[i][j][k][value];
                        }
                    }
                } else if (i == (int)Layer.size() - 1) {
                    for (int j = 0; j < Layer[i]; j++) {
                        load_fin >> b[i][j][value];
                    }
                } else {
                    for (int j = 0; j < Layer[i]; j++) {
                        for (int k = 0; k < Layer[i + 1]; k++) {
                            load_fin >> w[i][j][k][value];
                        }
                    }
                    for (int j = 0; j < Layer[i]; j++) {
                        load_fin >> b[i][j][value];
                    }
                }
            }
            weights_loaded = true;
        } else {
            cout << "无法打开参数文件: " << saving << endl;
        }
    }

    if (!weights_loaded) {
        for (int i = 0; i < (int)Layer.size() - 1; i++) {
            double sig = sqrt(2.0 / (Layer[i] + Layer[i + 1]));
            for (int j = 0; j < Layer[i]; j++) {
                for (int k = 0; k < Layer[i + 1]; k++) {
                    w[i][j][k][value] = get_random(sig);
                }
            }
        }
    }

    cout << "init net success\n";
}

void NET::save(string file) {
    ofstream fout(file, ios::out);
    for (int i = 0; i < (int)Layer.size(); i++) {
        if (i == 0) {
            for (int j = 0; j < Layer[i]; j++) {
                for (int k = 0; k < Layer[i + 1]; k++) {
                    fout << w[i][j][k][value] << endl;
                }
            }
        } else if (i == (int)Layer.size() - 1) {
            for (int j = 0; j < Layer[i]; j++) {
                fout << b[i][j][value] << endl;
            }
        } else {
            for (int j = 0; j < Layer[i]; j++) {
                for (int k = 0; k < Layer[i + 1]; k++) {
                    fout << w[i][j][k][value] << endl;
                }
            }
            for (int j = 0; j < Layer[i]; j++) {
                fout << b[i][j][value] << endl;
            }
        }
    }
}

void NET::Forward(TrainingSample& samp) {
    for (int i = 0; i < Layer[0]; i++) {
        v[0][i] = samp.in[i];
    }
    for (int i = 1; i < (int)Layer.size(); i++) {
        for (int j = 0; j < Layer[i]; j++) {
            v[i][j] = 0.0;
            for (int k = 0; k < Layer[i - 1]; k++) {
                v[i][j] += (v[i - 1][k] * w[i - 1][k][j][value]);
            }
            v[i][j] += b[i][j][value];
            v[i][j] = sigmoid(v[i][j]);
        }
    }
}

void NET::Backward(TrainingSample& samp) {
    for (int i = 0; i < Layer[(int)Layer.size() - 1]; i++) {
        b[(int)Layer.size() - 1][i][Temp] =
            (v[(int)Layer.size() - 1][i] - samp.out[i]) * v[(int)Layer.size() - 1][i] *
            (1 - v[(int)Layer.size() - 1][i]);
        b[(int)Layer.size() - 1][i][sum] += b[(int)Layer.size() - 1][i][Temp];
    }

    for (int i = (int)Layer.size() - 2; i > 0; i--) {
        for (int j = 0; j < Layer[i]; j++) {
            double temp = 0;
            for (int k = 0; k < Layer[i + 1]; k++) {
                temp += b[i + 1][k][Temp] * w[i][j][k][value];
                w[i][j][k][sum] += (b[i + 1][k][Temp] * v[i][j]);
            }
            b[i][j][Temp] = temp * v[i][j] * (1 - v[i][j]);
            b[i][j][sum] += b[i][j][Temp];
        }
    }

    for (int j = 0; j < Layer[0]; j++) {
        for (int k = 0; k < Layer[1]; k++) {
            w[0][j][k][sum] += (b[1][k][Temp] * v[0][j]);
        }
    }
}

void NET::Learning(int turn) {
    const double lr_decay = 0.998;
    vector<size_t> order;
    while (turn--) {
        order.resize(training_data.size());
        iota(order.begin(), order.end(), 0);
        shuffle(order.begin(), order.end(), gen);
        init();
        for (size_t idx : order) {
            Forward(training_data[idx]);
            Backward(training_data[idx]);
        }
        double data_size = (double)training_data.size();
        if (data_size <= 0.0) {
            return;
        }
        for (int j = 0; j < Layer[0]; j++) {
            for (int k = 0; k < Layer[1]; k++) {
                w[0][j][k][value] -= learning_rate * (w[0][j][k][sum] / data_size);
            }
        }
        for (int i = 1; i < (int)Layer.size() - 1; i++) {
            for (int j = 0; j < Layer[i]; j++) {
                b[i][j][value] -= learning_rate * (b[i][j][sum] / data_size);
                for (int k = 0; k < Layer[i + 1]; k++) {
                    w[i][j][k][value] -= learning_rate * (w[i][j][k][sum] / data_size);
                }
            }
        }
        for (int j = 0; j < Layer[(int)Layer.size() - 1]; j++) {
            b[(int)Layer.size() - 1][j][value] -=
                learning_rate * (b[(int)Layer.size() - 1][j][sum] / data_size);
        }
        learning_rate *= lr_decay;
    }
}

vector<vector<double>> NET::test() {
    vector<vector<double>> res;
    vector<double> temp;
    for (TrainingSample& samp : training_data) {
        Forward(samp);
        for (int i = 0; i < Layer[(int)Layer.size() - 1]; i++) {
            temp.push_back(v[(int)Layer.size() - 1][i]);
        }
        res.push_back(temp);
        vector<double>().swap(temp);
    }
    return res;
}

void NET::INIT() {
    init();
    for (int i = 0; i < (int)Layer.size() - 1; i++) {
        double sig = sqrt(2.0 / (Layer[i] + Layer[i + 1]));
        for (int j = 0; j < Layer[i]; j++) {
            for (int k = 0; k < Layer[i + 1]; k++) {
                w[i][j][k][value] = get_random(sig);
            }
        }
    }
    for (int i = 1; i < (int)Layer.size(); i++) {
        for (int j = 0; j < Layer[i]; j++) {
            b[i][j][value] = 0;
        }
    }
}

double NET::get_loss() {
    double loss = 0.0;
    if (training_data.empty()) {
        return loss;
    }
    for (TrainingSample& samp : training_data) {
        Forward(samp);
        for (int i = 0; i < Layer[(int)Layer.size() - 1]; i++) {
            loss += 0.5 * pow(v[(int)Layer.size() - 1][i] - samp.out[i], 2);
        }
    }
    loss /= (double)training_data.size();
    return loss;
}

void NET::init() {
    for (int i = 0; i < (int)Layer.size() - 1; i++) {
        for (int j = 0; j < Layer[i]; j++) {
            b[i][j][sum] = 0;
            for (int k = 0; k < Layer[i + 1]; k++) {
                w[i][j][k][sum] = 0;
            }
        }
    }
    for (int j = 0; j < Layer[(int)Layer.size() - 1]; j++) {
        b[(int)Layer.size() - 1][j][sum] = 0;
    }
}

double NET::get_random(double sig) {
    normal_distribution<double> dis(0, sig);
    return dis(gen);
}

double NET::sigmoid(double x) { return pow((1 + exp(-x)), -1); }

vector<double> NET::Predict(const vector<double>& input) {
    vector<double> result;
    if ((int)input.size() != InputSize()) {
        return result;
    }
    TrainingSample s;
    s.in = input;
    Forward(s);
    result.reserve(OutputSize());
    for (int i = 0; i < OutputSize(); i++) {
        result.push_back(v[(int)Layer.size() - 1][i]);
    }
    return result;
}

int NET::InputSize() const {
    if (Layer.empty()) {
        return 0;
    }
    return Layer.front();
}

int NET::OutputSize() const {
    if (Layer.empty()) {
        return 0;
    }
    return Layer.back();
}

int NET::GetSampleCount() const { return (int)training_data.size(); }

double NET::GetTrainAccuracy() {
    if (training_data.empty()) {
        return 0.0;
    }
    int correct = 0;
    const int out_n = Layer.back();
    for (TrainingSample& s : training_data) {
        Forward(s);
        int pred = 0;
        double best = v[(int)Layer.size() - 1][0];
        for (int i = 1; i < out_n; i++) {
            if (v[(int)Layer.size() - 1][i] > best) {
                best = v[(int)Layer.size() - 1][i];
                pred = i;
            }
        }
        int truth = 0;
        double best_y = s.out[0];
        for (int i = 1; i < out_n; i++) {
            if (s.out[i] > best_y) {
                best_y = s.out[i];
                truth = i;
            }
        }
        if (pred == truth) {
            correct++;
        }
    }
    return (double)correct / (double)training_data.size();
}
