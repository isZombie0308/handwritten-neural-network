#include "class.hpp"
#include <filesystem>
#include <iostream>
#include <memory>
using namespace std;

int main() {
	std::filesystem::create_directories("file");

	std::unique_ptr<NET> net;
	std::unique_ptr<NET> trainer;
	bool training_in_progress = false;
	int train_epoch = 0;
	const int kTotalEpoch = 300;
	const std::vector<int> layers{ kGridSize * kGridSize, 256, 10 };
	const double kLearnRate = 0.025;
	const std::string training_file = "file\\training_digits.txt";
	// 隐藏层改为 256 后须用新文件名，勿与旧 128 维权重混用
	const std::string saving_file = "file\\saving_digits_256.txt";
	net = std::make_unique<NET>(kLearnRate, layers, training_file, saving_file);
	attach_network(net.get());
	fout.open("file\\out.txt", ios::out);
	if (fout.is_open() == 0) {
		cout << "failed to open file\\out.txt";
		return 0;
	}

	initgraph(kWindowWidth, kWindowHeight);
	End = false;
	click = false;
	temp_click = false;
	clear_canvas();
	init_ui();
	set_status_text(L"Ready");

	BeginBatchDraw();

	while (End==0) {
		get_message();
		logic();
		if (g_request_train && !training_in_progress) {
			g_request_train = false;
			trainer = std::make_unique<NET>(kLearnRate, layers, training_file);
			if (trainer->GetSampleCount() == 0) {
				set_status_text(L"No training samples! Draw and save first.");
				trainer.reset();
			}
			else {
				training_in_progress = true;
				train_epoch = 0;
				set_status_text(L"Training... 0/" + std::to_wstring(kTotalEpoch));
			}
		}

		if (training_in_progress && trainer) {
			trainer->Learning(1);
			train_epoch++;
			if (train_epoch % 5 == 0 || train_epoch == kTotalEpoch) {
				const double loss = trainer->get_loss();
				set_status_text(L"Training... " + std::to_wstring(train_epoch) + L"/" +
					std::to_wstring(kTotalEpoch) + L" loss=" + std::to_wstring(loss));
			}
			if (train_epoch >= kTotalEpoch) {
				const double loss = trainer->get_loss();
				const double acc = trainer->GetTrainAccuracy();
				trainer->save(saving_file);
				net = std::make_unique<NET>(kLearnRate, layers, training_file, saving_file);
				attach_network(net.get());
				training_in_progress = false;
				trainer.reset();
				set_status_text(L"Done acc=" + std::to_wstring(acc) + L" loss=" + std::to_wstring(loss) + L" saved");
			}
		}
		Draw();
	}

	EndBatchDraw();
	closegraph();
	fout.close();
	
}