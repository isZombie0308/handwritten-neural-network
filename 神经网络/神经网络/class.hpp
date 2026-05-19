#pragma once

#include <easyx.h>
#include <fstream>
#include <string>
#include <vector>

#include "net.hpp"
constexpr int kGridSize = 35;
constexpr int kCanvasPixels = 400;
constexpr int kWindowWidth = 400;
constexpr int kWindowHeight = 600;

extern std::ofstream fout;
extern ExMessage msg;
extern bool End;
extern bool click;
extern bool temp_click;
extern int canvas[kGridSize][kGridSize];
extern int g_pred_digit;
extern int g_last_label;
extern bool g_request_train;

class B {
public:
    bool in = false;
    int top = 0;
    int bot = 0;
    int left = 0;
    int right = 0;

    void update(ExMessage& msg);
    void draw(const wchar_t* label) const;
};

void get_message();
bool is_in_circle(int i, int j, int x, int y, double r);
void draw(int x, int y);
void logic();
void Draw();
void init_ui();
void clear_canvas();
bool append_training_sample(const int (&canvas)[kGridSize][kGridSize], int label, const char* path);
void set_status_text(const std::wstring& text);

void attach_network(NET* net);
int predict_digit(const int (&canvas)[kGridSize][kGridSize], std::vector<double>* probs = nullptr);