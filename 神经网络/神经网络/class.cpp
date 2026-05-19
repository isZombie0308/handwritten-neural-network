#include "class.hpp"

#include <algorithm>
#include <string>

using namespace std;

ofstream fout;
ExMessage msg;
IMAGE img[11];
bool End;
bool click;
bool temp_click;
int canvas[kGridSize][kGridSize] = {0};
int g_pred_digit = -1;
int g_last_label = -1;
bool g_request_train = false;

static NET* g_net = nullptr;
static B g_predict_btn;
static B g_save_btn;
static B g_clear_btn;
static B g_quit_btn;
static bool g_mouse_down = false;
static int g_pressed_button = -1;
static int g_clicked_button = -1;
static wstring g_status_text = L"Ready";
static bool has_ink_pixels(const int (&canvas_ref)[kGridSize][kGridSize]) {
    for (int j = 0; j < kGridSize; j++) {
        for (int i = 0; i < kGridSize; i++) {
            if (canvas_ref[i][j] > 0) {
                return true;
            }
        }
    }
    return false;
}

static int get_button_id(int x, int y) {
    if (x >= g_predict_btn.left && x <= g_predict_btn.right && y >= g_predict_btn.top &&
        y <= g_predict_btn.bot) {
        return 0;
    }
    if (x >= g_save_btn.left && x <= g_save_btn.right && y >= g_save_btn.top &&
        y <= g_save_btn.bot) {
        return 1;
    }
    if (x >= g_clear_btn.left && x <= g_clear_btn.right && y >= g_clear_btn.top &&
        y <= g_clear_btn.bot) {
        return 2;
    }
    if (x >= g_quit_btn.left && x <= g_quit_btn.right && y >= g_quit_btn.top &&
        y <= g_quit_btn.bot) {
        return 3;
    }
    return -1;
}

void load() {
    const int cell = max(1, kCanvasPixels / kGridSize);
    loadimage(&img[0], L"image\\bk.png", kWindowWidth, kWindowHeight);
    loadimage(&img[1], L"image\\brush.png", cell, cell);
    constexpr int btn_w = 92;
    constexpr int btn_h = 45;
    loadimage(&img[2], L"image\\yes.png", btn_w, btn_h);
    loadimage(&img[3], L"image\\yes1.png", btn_w, btn_h);
    loadimage(&img[4], L"image\\quit.png", btn_w, btn_h);
    loadimage(&img[5], L"image\\quit1.png", btn_w, btn_h);
    loadimage(&img[6], L"image\\save.png", btn_w, btn_h);
    loadimage(&img[7], L"image\\save1.png", btn_w, btn_h);
    loadimage(&img[8], L"image\\clear.png", btn_w, btn_h);
    loadimage(&img[9], L"image\\clear1.png", btn_w, btn_h);
    loadimage(&img[10], L"image\\round.png", cell, cell);
}

void B::update(ExMessage& msg_ref) {
    in = false;
    if (msg_ref.x >= left && msg_ref.y >= top && msg_ref.x <= right && msg_ref.y <= bot) {
        in = true;
    }
}

void B::draw(const wchar_t* label) const {
    setlinecolor(BLACK);
    setfillcolor(in ? RGB(190, 230, 255) : RGB(235, 235, 235));
    solidrectangle(left, top, right, bot);
    settextcolor(BLACK);
    setbkmode(TRANSPARENT);
    outtextxy(left + 12, top + 10, label);
}

void get_message() {
    temp_click = false;
    g_clicked_button = -1;
    while (peekmessage(&msg)) {
        switch (msg.message) {
            case WM_MOUSEMOVE:
                break;
            case WM_LBUTTONDOWN:
                click = true;
                g_mouse_down = true;
                g_pressed_button = get_button_id(msg.x, msg.y);
                break;
            case WM_LBUTTONUP:
                click = false;
                if (g_mouse_down) {
                    const int release_button = get_button_id(msg.x, msg.y);
                    if (release_button >= 0 && release_button == g_pressed_button) {
                        g_clicked_button = release_button;
                        temp_click = true;
                    }
                }
                g_mouse_down = false;
                g_pressed_button = -1;
                break;
            case WM_CHAR:
                if (msg.ch >= '0' && msg.ch <= '9') {
                    g_last_label = (int)(msg.ch - '0');
                } else if (msg.ch == 't' || msg.ch == 'T') {
                    g_request_train = true;
                }
                break;
            case WM_CLOSE:
                End = true;
                break;
            default:
                break;
        }
    }
}

bool is_in_circle(int i, int j, int x, int y, double r) {
    if ((i - x) * (i - x) + (j - y) * (j - y) < r * r) {
        return true;
    }
    return false;
}

void draw(int x, int y) {
    for (int i = x - 2; i < x + 3; i++) {
        for (int j = y - 2; j < y + 3; j++) {
            if (i >= 0 && i < kGridSize && j >= 0 && j < kGridSize) {
                if (is_in_circle(x, y, i, j, 1.5)) {
                    canvas[i][j] = 2;
                } else if (is_in_circle(x, y, i, j, 2.1) && canvas[i][j] != 2) {
                    canvas[i][j] = 1;
                }
            }
        }
    }
}

void logic() {
    g_predict_btn.update(msg);
    g_save_btn.update(msg);
    g_clear_btn.update(msg);
    g_quit_btn.update(msg);

    if (click && msg.x >= 0 && msg.x < kCanvasPixels && msg.y >= 0 && msg.y < kCanvasPixels) {
        const int grid = max(1, kCanvasPixels / kGridSize);
        const int gx = min(kGridSize - 1, max(0, msg.x / grid));
        const int gy = min(kGridSize - 1, max(0, msg.y / grid));
        draw(gx, gy);
    }

    if (temp_click) {
        if (g_clicked_button == 3) {
            End = true;
            return;
        }
        if (g_clicked_button == 2) {
            clear_canvas();
            g_pred_digit = -1;
            g_status_text = L"Canvas cleared";
            return;
        }
        if (g_clicked_button == 1) {
            if (g_last_label >= 0 && g_last_label <= 9) {
                if (!has_ink_pixels(canvas)) {
                    g_status_text = L"Canvas is empty";
                } else {
                    const bool ok = append_training_sample(canvas, g_last_label, "file\\training_digits.txt");
                    g_status_text = ok ? L"Sample saved" : L"Cannot open training file";
                }
            } else {
                g_status_text = L"Label not set (press 0-9)";
            }
            return;
        }
        if (g_clicked_button == 0) {
            vector<double> probs;
            g_pred_digit = predict_digit(canvas, &probs);
            g_status_text = (g_pred_digit >= 0) ? L"Predict done" : L"Predict failed";
            if (fout.is_open()) {
                fout << "predict=" << g_pred_digit << "\n";
                for (int j = 0; j < kGridSize; j++) {
                    for (int i = 0; i < kGridSize; i++) {
                        fout << canvas[i][j] << ' ';
                    }
                    fout << '\n';
                }
                fout << "---\n";
            }
        }
    }
}

void Draw() {
    cleardevice();
    putimage(0, 0, &img[0]);

    setlinecolor(RGB(220, 220, 220));
    rectangle(0, 0, kCanvasPixels, kCanvasPixels);

    const int grid = max(1, kCanvasPixels / kGridSize);
    for (int i = 0; i < kGridSize; i++) {
        for (int j = 0; j < kGridSize; j++) {
            if (canvas[i][j] == 2) {
                putimage(i * grid, j * grid, &img[1]);
            } else if (canvas[i][j] == 1) {
                putimage(i * grid, j * grid, &img[10]);
            }
        }
    }

    putimage(g_predict_btn.left, g_predict_btn.top, g_predict_btn.in ? &img[3] : &img[2]);
    putimage(g_save_btn.left, g_save_btn.top, g_save_btn.in ? &img[7] : &img[6]);
    putimage(g_clear_btn.left, g_clear_btn.top, g_clear_btn.in ? &img[9] : &img[8]);
    putimage(g_quit_btn.left, g_quit_btn.top, g_quit_btn.in ? &img[5] : &img[4]);

    setbkmode(TRANSPARENT);
    settextcolor(BLACK);
    if (g_pred_digit >= 0) {
        wstring text = L"Result: " + to_wstring(g_pred_digit);
        outtextxy(20, 560, text.c_str());
    } else {
        outtextxy(20, 560, L"Result: N/A");
    }
    outtextxy(20, 535, g_status_text.c_str());
    outtextxy(200, 560, L"0-9:set label, T:train");
    FlushBatchDraw();
}

void attach_network(NET* net) { g_net = net; }

int predict_digit(const int (&canvas)[kGridSize][kGridSize], vector<double>* probs) {
    if (g_net == nullptr) {
        return -1;
    }

    vector<double> input;
    input.reserve(kGridSize * kGridSize);
    for (int j = 0; j < kGridSize; j++) {
        for (int i = 0; i < kGridSize; i++) {
            input.push_back(canvas[i][j] > 0 ? 1.0 : 0.0);
        }
    }

    if ((int)input.size() != g_net->InputSize()) {
        return -1;
    }

    vector<double> out = g_net->Predict(input);
    if (out.empty()) {
        return -1;
    }

    if (probs != nullptr) {
        *probs = out;
    }

    int best_idx = 0;
    double best_val = out[0];
    for (int i = 1; i < (int)out.size(); i++) {
        if (out[i] > best_val) {
            best_val = out[i];
            best_idx = i;
        }
    }
    return best_idx;
}

void init_ui() {
    load();
    g_predict_btn.left = 8;
    g_predict_btn.right = 100;
    g_predict_btn.top = 430;
    g_predict_btn.bot = 475;

    g_save_btn.left = 104;
    g_save_btn.right = 196;
    g_save_btn.top = 430;
    g_save_btn.bot = 475;

    g_clear_btn.left = 200;
    g_clear_btn.right = 292;
    g_clear_btn.top = 430;
    g_clear_btn.bot = 475;

    g_quit_btn.left = 296;
    g_quit_btn.right = 392;
    g_quit_btn.top = 430;
    g_quit_btn.bot = 475;
}

void clear_canvas() {
    for (int i = 0; i < kGridSize; i++) {
        for (int j = 0; j < kGridSize; j++) {
            canvas[i][j] = 0;
        }
    }
}

bool append_training_sample(const int (&canvas_ref)[kGridSize][kGridSize], int label, const char* path) {
    if (label < 0 || label > 9) {
        return false;
    }
    bool has_ink = false;
    for (int j = 0; j < kGridSize; j++) {
        for (int i = 0; i < kGridSize; i++) {
            if (canvas_ref[i][j] > 0) {
                has_ink = true;
                break;
            }
        }
        if (has_ink) {
            break;
        }
    }
    if (!has_ink) {
        return false;
    }

    ofstream out(path, ios::app);
    if (!out.is_open()) {
        return false;
    }

    // 1225 inputs
    for (int j = 0; j < kGridSize; j++) {
        for (int i = 0; i < kGridSize; i++) {
            out << (canvas_ref[i][j] > 0 ? 1 : 0) << ' ';
        }
    }

    // 10-d one-hot
    for (int k = 0; k < 10; k++) {
        out << (k == label ? 1 : 0) << (k == 9 ? '\n' : ' ');
    }
    return true;
}

void set_status_text(const wstring& text) { g_status_text = text; }
