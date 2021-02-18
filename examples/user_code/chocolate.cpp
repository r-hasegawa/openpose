// #define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY

// Command-line user intraface
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

#include <iostream>
#include <fstream>

// Custom OpenPose flags
// Producer
// DEFINE_string(image_dir,                "/data/input/0531A4-1_4k.mp4",
//     "Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).");
// Display
// DEFINE_bool(no_display, false,
//     "Enable to disable the visual display.");

static int busshoku_count = 0;

std::vector<std::string> split(std::string str, char del) {
    int first = 0;
    int last = str.find_first_of(del);
 
    std::vector<std::string> result;
 
    while (first < str.size()) {
        std::string subStr(str, first, last - first);
 
        result.push_back(subStr);
 
        first = last + 1;
        last = str.find_first_of(del, first);
 
        if (last == std::string::npos) {
            last = str.size();
        }
    }
 
    return result;
}

template <typename _Ty>
std::ostream& operator << (std::ostream& ostr, const std::vector<_Ty>& v) {
    if (v.empty()) {
        ostr << "{ }";
        return ostr;
    }
    ostr << "{" << v.front();
    for (auto itr = ++v.begin(); itr != v.end(); itr++) {
        ostr << ", " << *itr;
    }
    ostr << "}";
    return ostr;
}


int vector_finder(std::vector<int> vec, int number) {
  auto itr = std::find(vec.begin(), vec.end(), number);
  size_t index = std::distance( vec.begin(), itr );
  if (index != vec.size()) { // 発見できたとき
    return std::distance(vec.begin(), itr);
  }
  else { // 発見できなかったとき
    return -1;
  }
}

int printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr, int tl_x, int tl_y, int br_x, int br_y, int arr[])
{
    // std::cout << fnum << std::endl;
    // 1.Nose 2.Chest 3.RShoulder 4.RElbow 5.RWrist
    // 6.LShoulder 7.LElbow 8.LWrist 9.MidHip 10.RHip
    // 11.RKnee 12.RAnkle 13.LHip 14.LKnee 15.LAnkle
    // 16. REye 17.LEye 18.REar 19.LEar 20.Neck 21.Head
    const int key_id = 2;
    const int continuance_threshold = 1; // 連続検出フレーム数のしきい値

    try
    {
        // Example: How to use the pose keypoints
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            cv::Mat disp_image = datumsPtr->at(0)->cvOutputData;
            auto poseIds = datumsPtr->at(0)->poseIds;
            auto poseKeypoints = datumsPtr->at(0)->poseKeypoints;

            bool is_busshoku = false;
            int face_max_x, face_max_y, face_min_x, face_min_y;
            int face_offset = 100;
            int face_key_ids[] = {1, 16, 17, 18, 19, 20, 21};
            cv::rectangle(disp_image, cv::Point(tl_x, tl_y),  cv::Point(br_x, br_y), cv::Scalar::all(255), 3, cv::LINE_AA);
            // cv::putText(disp_image, "hello", cv::Point(20,20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,255,255), 3, CV_AA);

            for(int i = 0; i < poseKeypoints.getSize(0); i++){
                if(((tl_x <= poseKeypoints[i*21*3+5*3-3]) && (poseKeypoints[i*21*3+5*3-3] <= br_x)
                && (tl_y <= poseKeypoints[i*21*3+5*3-2]) && (poseKeypoints[i*21*3+5*3-2] <= br_y))
                || ((tl_x <= poseKeypoints[i*21*3+8*3-3]) && (poseKeypoints[i*21*3+8*3-3] <= br_x)
                && (tl_y <= poseKeypoints[i*21*3+8*3-2]) && (poseKeypoints[i*21*3+8*3-2] <= br_y)))
                {
                    is_busshoku = true;
                    face_max_x = 0;
                    face_min_x = disp_image.cols;
                    face_max_y = 0;
                    face_min_y = disp_image.rows;
                    for (int j = 0; j < 7; j++){
                        if(poseKeypoints[i*21*3+face_key_ids[j]*3-1] > 0.0){
                            // update min/max face area
                            if(poseKeypoints[i*21*3+face_key_ids[j]*3-3] > face_max_x)
                                face_max_x = poseKeypoints[i*21*3+face_key_ids[j]*3-3] + face_offset;
                            if(poseKeypoints[i*21*3+face_key_ids[j]*3-3] < face_min_x)
                                face_min_x = poseKeypoints[i*21*3+face_key_ids[j]*3-3] - face_offset;
                            if(poseKeypoints[i*21*3+face_key_ids[j]*3-2] > face_max_y)
                                face_max_y = poseKeypoints[i*21*3+face_key_ids[j]*3-2] + face_offset;
                            if(poseKeypoints[i*21*3+face_key_ids[j]*3-2] < face_min_y)
                                face_min_y = poseKeypoints[i*21*3+face_key_ids[j]*3-2] - face_offset;
                        }
                    }
                    if(face_max_x > disp_image.cols)
                        face_max_x = disp_image.cols;
                    if(face_min_x < 0)
                        face_min_x = 0;
                    if(face_max_y > disp_image.rows)
                        face_max_y = disp_image.rows;
                    if(face_min_y < 0)
                        face_min_y = 0;
                    // face_max_x, face_max_y, face_min_x, face_min_y;
                    arr[0] = face_min_x;
                    arr[1] = face_min_y;
                    arr[2] = face_max_x - face_min_x;
                    arr[3] = face_max_y - face_min_y;
                    cv::putText(disp_image, "Someone is interested in CHOCOLATE !!", cv::Point(50,70), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0,0,255), 5, CV_AA);
                }
            }

            if (is_busshoku){
                busshoku_count = busshoku_count + 1;
            }else{
                busshoku_count = 0;
            }
            //poseKeypoints[i*21*3+key_id*3-1]; //2:nose 5:chest

            // 表示
            if (true){
                if (disp_image.cols > 1500){
                    cv::resize(disp_image, disp_image, cv::Size(), 0.6, 0.6);
                }
                cv::imshow("チョコレート監視カメラ", disp_image);
            }
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
        const auto key = (char)cv::waitKey(1);
        if (key == 27){
            return 1;
        }else{
            if (busshoku_count >= 15){
                busshoku_count = 0;
                return 2;
            }
            return 0;
        }
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return 1;
    }
}

struct ImageInfo {
    cv::Mat img;       // 入力画像と出力画像
    cv::Point Pt[2];   // 変換前の4頂点座標（左上, 右上, 右下, 左下）
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
        std::cout << "info.Pt[" << info.pos << "] = {" << x << "," << y << "};"<< std::endl;
        // 左ボタンを押したとき、4頂点のうち一番近い点を探す
        if (info.pos < 2)
        {
            info.Pt[info.pos] = p;
            info.pos++;
            cv::circle(info.img, cv::Point(x, y), 3, (0, 0, 255), 3);
            cv::imshow(info.winName, info.img);
        }
        else{
            std::cout << "Error: Already defined 4 points" << std::endl;
        }
        if (info.pos == 2){
        	std::cout << "Please press any key." << std::endl;
			cv::rectangle(info.img,info.Pt[0], info.Pt[1], cv::Scalar::all(255), 3, cv::LINE_AA);
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

void getM (cv::Mat frame, int arr[])//変換行列の作成
{
	// 射影変換のための画像情報構造体
    ImageInfo info;
    // 画像を読み込む
    info.img = frame;

    // コールバック関数を登録する
    info.winName = "test";
    cv::namedWindow(info.winName);
    cv::setMouseCallback(info.winName, mouseCallback, (void *)&info);
    cv::imshow(info.winName, info.img);
    cv::waitKey();
    cv::destroyAllWindows();
    arr[0] = info.Pt[0].x;
    arr[1] = info.Pt[0].y;
    arr[2] = info.Pt[1].x;
    arr[3] = info.Pt[1].y;
}


void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        // Configuring OpenPose

        // logging_level
        op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                  __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // producerType
        op::ProducerType producerType;
        std::string producerString;
        std::tie(producerType, producerString) = op::flagsToProducer(
            FLAGS_image_dir, FLAGS_video, FLAGS_ip_camera, FLAGS_camera, FLAGS_flir_camera, FLAGS_flir_camera_index);
        // cameraSize
        const auto cameraSize = op::flagsToPoint(FLAGS_camera_resolution, "-1x-1");
        // outputSize
        const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "1312x736");
        // const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
        // -1x368 -> (16:9 = 1920x1080) -> 656x368
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(FLAGS_face_net_resolution, "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(FLAGS_hand_net_resolution, "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::log("Flag `write_keypoint` is deprecated and will eventually be removed."
                    " Please, use `write_json` instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1 || FLAGS_flir_camera);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
            FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
            poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
            FLAGS_part_to_show, FLAGS_model_folder, heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
            (float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
            FLAGS_prototxt_path, FLAGS_caffemodel_path, (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold};
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold};
        opWrapper.configure(wrapperStructHand);
        // Tracking (use op::WrapperStructTracking{} to disable it)
        const op::WrapperStructTracking wrapperStructTracking{
            FLAGS_tracking}; // Raaj: Add your flags in here
        opWrapper.configure(wrapperStructTracking);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, -1, FLAGS_ik_threads};
        opWrapper.configure(wrapperStructExtra);
        // Producer (use default to disable any input)
        // const op::WrapperStructInput wrapperStructInput{
        //     producerType, producerString, FLAGS_frame_first, FLAGS_frame_step, FLAGS_frame_last,
        //     FLAGS_process_real_time, FLAGS_frame_flip, FLAGS_frame_rotate, FLAGS_frames_repeat,
        //     cameraSize, FLAGS_camera_parameter_path, FLAGS_frame_undistort, FLAGS_3d_views};
        // opWrapper.configure(wrapperStructInput);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, FLAGS_write_keypoint, op::stringToDataFormat(FLAGS_write_keypoint_format),
            FLAGS_write_json, FLAGS_write_coco_json, FLAGS_write_coco_json_variants, FLAGS_write_coco_json_variant,
            FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video, FLAGS_write_video_fps,
            FLAGS_write_video_with_audio, FLAGS_write_heatmaps, FLAGS_write_heatmaps_format, FLAGS_write_video_3d,
            FLAGS_write_video_adam, FLAGS_write_bvh, FLAGS_udp_host, FLAGS_udp_port};
        opWrapper.configure(wrapperStructOutput);
        // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
        // opWrapper.configure(op::WrapperStructGui{});
        // // GUI (comment or use default argument to disable any visual output)
        // const op::WrapperStructGui wrapperStructGui{
        //     op::flagsToDisplayMode(FLAGS_display, FLAGS_3d), !FLAGS_no_gui_verbose, FLAGS_fullscreen};
        // opWrapper.configure(wrapperStructGui);
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int openPoseDemo()
{
    try
    {
        // Configuring OpenPose
        op::log("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        configureWrapper(opWrapper);
        // Starting OpenPose
        op::log("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        cv::Mat frame, face_image; //取得したフレーム
        cv::VideoCapture cap(FLAGS_video);
        if(!cap.isOpened())//カメラデバイスが正常にオープンしたか確認．
        {
            std::cout << "Input Error" << std::endl;
            return -1;
        }

        cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920); // カメラ画像の横幅を1280に設定
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080); // カメラ画像の縦幅を720に設定
        for(int i = 0;i<10;i++){
        cap.read(frame);
        }

        auto nowms = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(nowms);
        std::stringstream sss;
        sss << std::put_time(localtime(&now_c), "%Y%m%d_%H%M%S");
        std::cout << sss.str() << std::endl;



        std::vector<int> zoom(4);

        int  choco[] = {0, 0, 0, 0};
        cv::flip(frame, frame, 1); // 水平反転
        getM(frame, choco);
        std::cout << choco[0] << choco[1] <<  choco[2] << choco[3] << std::endl;


        int criminal[] = {0, 0, 100, 100};



        op::log("Starting OpenPose demo...", op::Priority::High);
        int fnum = 0;
        cap.set(CV_CAP_PROP_POS_FRAMES,0);
        while(cap.read(frame))//無限ループ
        {
            cv::flip(frame, frame, 1); // 水平反転
        	if (frame.cols > 1920){
        		cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
        	}
            const auto imageToProcess = frame;

        // // Process and display images
            auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
            if (datumProcessed != nullptr)
            {
                // printKeypoints(datumProcessed, bg, M, offset);

                const auto ret = printKeypoints(datumProcessed, choco[0], choco[1],  choco[2], choco[3], criminal);
                if (ret == 1)
                {
                    op::log("User pressed Esc to exit demo.", op::Priority::High);
                    fnum++;
                    break;
                }
                else if(ret == 2)
                {
                    // show criminal image
                    // std::cout << std::endl
                    cv::Rect roi(cv::Point(criminal[0], criminal[1]), cv::Size(criminal[2], criminal[3]));
                    cv::resize(frame(roi), face_image, cv::Size(), 1.5, 1.5);
                    cv::imshow("最近物色した人物", face_image);
                }
            }
            else{
                // op::log("Image " + imagePath + " could not be processed.", op::Priority::High);
                op::log("Video could not be processed.", op::Priority::High);
            }
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        return -1;
    }
}


int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running openPoseDemo
    return openPoseDemo();
}
