#include "RELUnet.cpp"
#include <SFML/Graphics.hpp>
void getInput(sf::Image& pic, std::vector<float>& input) {
    int width = pic.getSize().x;
    int height = pic.getSize().y;
    int tmp;
    sf::Color color;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            color = pic.getPixel(i, j);
            tmp = color.r + color.g + color.b;
            input[i * height + j] = tmp * 1.0 / 765.0;
        }
    }
}
int IndexOfMax(std::vector<float>& answers) {
    int n = answers.size();
    float a = 0;
    float max = 0;
    for (int i = 0; i < n; i++) {
        if (answers[i] > a) {
            a = answers[i];
            max = i;
        }
    }
    return max;
}
int main()
{
    int epochs_count = 4;
    std::vector<int> config = { 784, 800, 10 };
    std::string PathToWeights = "MnistWeights";
    Net net(config, PathToWeights);
    std::vector<float> answer(10, 0);
    std::vector<float> input(784, 0);
    std::vector<float> networkOutput(10, 0);
    std::string PicFolder = "train/";
    std::string name;
    std::string mode;
    float LearningRate = 0.0005;
    int k = 0;
    int tensCount = 0;

    std::cout << "Enter mode: test/train\n";
    std::cin >> mode;
    if (mode == "learn" || mode == "train") {
        std::ofstream logfile("logfile.txt");
        logfile.precision(2);
        for (int l = 0; l < epochs_count; l++) {

            std::vector<std::ifstream> list(10);
            std::string path;
            for (int j = 0; j < 10; j++) {
                path = "lists/trainlist";
                path += ('0' + j);
                path += ".txt";
                list[j].open(path);
                if (list[j].good()) {
                    std::cout << path <<" is good\n";
                }
                else {
                    std::cout << path << " failed to open\n";
                }
            }
            int TestCount = 0;
            float loss = 0.0f;
            float total_loss = 0.0f;
            while (TestCount < 37900) {
                tensCount++;
                total_loss = 0.0f;
                for (int i = 0; i < 10; i++) {
                    answer.assign(10, 0.0);
                    answer[i] = 1.0;
                    if (list[i] >> name) {
                        sf::Image pic;
                        if (pic.loadFromFile(PicFolder + name)) {
                            k++;
                            //std::cout<<"test #"<<k<<": loaded " <<PicFolder+name<<"\n";
                            getInput(pic, input);
                            networkOutput = net.forward(input);
                            loss = net.backprop(answer, LearningRate);
                            logfile << "test #" << k  << " target: " << i  << ",answer: " << IndexOfMax(networkOutput) << ", loss: " << loss << ", file " << name << "\n";
                            logfile << "probabilities: ";
                            for (int j = 0; j < config.back(); j++)
                                logfile << j << ": " << networkOutput[j] << " " << std::fixed;
                            logfile << "\n";
                            total_loss += loss;
                        }
                    }
                    else{
                        std::cout << "Failed to load " << i << "\n";
                    }
                    TestCount++;
                }
                std::cout << "loaded 10 pics(" << tensCount << "), loss is " << total_loss << "\n";
            }
            LearningRate *= 0.4;

            std::cout << "saving weights...\n";
            net.SaveWeightsToFile(PathToWeights);
        }
    }
    if (mode == "test") {
        sf::Image pic;
        std::vector<int> RightAnswers(10, 0);
        std::vector<int> WrongAnswers(10, 0);
        std::string PicName;
        std::string PathToPic = "test/";
        std::string path;
        int TotalRight = 0;
        int TotalWrong = 0;
        for (int i = 0; i < 10; i++) {
            path = "lists/testlist";
            path += ('0' + i);
            path += ".txt";
            std::ifstream file(path);
            int passes = 0;
            int passes_per_num = 100;
            while (file >> PicName && passes < passes_per_num) {
                if (pic.loadFromFile(PathToPic + PicName)) {
                    getInput(pic, input);
                    networkOutput = net.forward(input);
                    std::cout << "Loaded " << PathToPic + PicName << "\n";
                    int number = IndexOfMax(networkOutput);
                    if (number == i) {
                        RightAnswers[i]++;
                        TotalRight++;
                    }
                    else {
                        WrongAnswers[i]++;
                        TotalWrong++;
                    }
                }
                ++passes_per_num;
            }
        }
        int total;
        for (int i = 0; i < 10; i++) {
            std::cout << "Number " << i << ":\n";
            total = WrongAnswers[i] + RightAnswers[i];
            std::cout << "Guessed: " << RightAnswers[i] << '/' << total << " Failed to guess: " << WrongAnswers[i] << '/' << total << " Accuracy is " << 100 * RightAnswers[i] * 1.0 / (total * 1.0) << "%\n";
        }
        total = TotalRight + TotalWrong;
        std::cout << "Total right answers: " << TotalRight << "\n"
             << "Total wrong answers: " << TotalWrong << "\nAccuracy is " << 100 * TotalRight * 1.0 / (total * 1.0) << "%\n";
    }
}
