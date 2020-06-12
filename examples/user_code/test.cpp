#include <opencv2/opencv.hpp>

// 射影変換のための各情報を保存する構造体
struct ImageInfo {
    cv::Mat img;       // 入力画像と出力画像
    cv::Point2f Pt[4];   // 変換前の4頂点座標（左上, 右上, 右下, 左下）
    std::string winName;    // 出力ウインドウの名前
    int pos = 0;
};

// コールバック関数
void mouseCallback(int event, int x, int y, int flags, void *data)
{
    static int select = -1;     // マウスで選択された頂点番号（-1:選択無し）
    cv::Point2f p(x, y);        // マウスの座標
    double dis = 1e10;

    ImageInfo &info = *(ImageInfo *)data;

    switch (event) {
    case cv::EVENT_LBUTTONDOWN:
        // 左ボタンを押したとき、4頂点のうち一番近い点を探す
        if (info.pos < 4)
        {
            info.Pt[info.pos] = p;
            info.pos++;
            cv::circle(info.img, cv::Point(x, y), 3, (0, 0, 255), 3);
            cv::imshow(info.winName, info.img);
        }
        else{
            std::cout << "Error: Already defined 4 points" << std::endl;
        }

        break;

    case cv::EVENT_MOUSEMOVE:
        // マウスの移動
        cv::Mat img2 = info.img.clone();
        int h = img2.rows;
        int w = img2.cols;
        cv::line(img2, cv::Point(x, 0), cv::Point(x, h-1), (255, 0, 0));
        cv::line(img2, cv::Point(0, y), cv::Point(w-1, y), (255, 0, 0));
        cv::imshow(info.winName, img2);
        break;

    }
}

int main(void)
{
    // 射影変換のための画像情報構造体
    ImageInfo info;

    // 画像を読み込む
    info.img = cv::imread("./test.png", cv::IMREAD_COLOR);

    // コールバック関数を登録する
    info.winName = "test";
    cv::namedWindow(info.winName);
    cv::setMouseCallback(info.winName, mouseCallback, (void *)&info);
    cv::imshow(info.winName, info.img);
    cv::waitKey();
    cv::destroyAllWindows();


    return 0;
}
