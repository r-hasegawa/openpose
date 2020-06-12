// #define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY

// Command-line user intraface
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

// Custom OpenPose flags
// Producer
// DEFINE_string(image_dir,                "/data/input/0531A4-1_4k.mp4",
//     "Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).");
// Display
DEFINE_bool(no_display, false,
    "Enable to disable the visual display.");

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


bool printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr, 
	cv::Mat bg, cv::Mat M, cv::Point offset, int area_resize_rate,
	cv::VideoWriter writer1, cv::VideoWriter writer2)
{
	// 1.Nose 2.Chest 3.RShoulder 4.RElbow 5.RWrist
	// 6.LShoulder 7.LElbow 8.LWrist 9.MidHip 10.RHip
	// 11.RKnee 12.RAnkle 13.LHip 14.LKnee 15.LAnkle
	// 16. REye 17.LEye 18.REar 19.LEar 20.Neck 21.Head

    try
    {
        // Example: How to use the pose keypoints
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
        	cv::Mat disp_image = datumsPtr->at(0)->cvOutputData;
        	if (disp_image.cols > 1500){
        		cv::resize(disp_image, disp_image, cv::Size(), 0.5, 0.5);
        	}
			cv::imshow("OpenPose Tracking", disp_image);
			writer1 << disp_image;

        	auto poseIds = datumsPtr->at(0)->poseIds;
			auto poseKeypoints = datumsPtr->at(0)->poseKeypoints;
    		// ワールド座標系へ変換
    		// cv::Point points[poseKeypoints.getSize(0)];
    		// cv::Point pos[poseKeypoints.getSize(0)];
    		cv::Mat bg_copy = bg.clone();
    		std::vector<cv::Point2f> points(poseKeypoints.getSize(0));
			std::vector<cv::Point2f> positions(poseKeypoints.getSize(0));
			for(int i = 0; i < poseKeypoints.getSize(0); i++){
				const auto id = int (poseIds[i]);
				const int key_id = 12;
				const float Keypoint = poseKeypoints[i*21*3+key_id*3-1]; //2:nose 5:chest
				// std::cout << id << std::endl;
				if (Keypoint < 0.05)
					continue;
				points[i] = cv::Point( poseKeypoints[i*21*3+key_id*3-3], poseKeypoints[i*21*3+key_id*3-2] );
			}
			if ( poseKeypoints.getSize(0) > 0){ // ワールド座標へ変換
				cv::perspectiveTransform(points, positions, M);
			}
			// 描画処理
			std::vector<cv::Point> pos(poseKeypoints.getSize(0));
			for(int i = 0; i < poseKeypoints.getSize(0); i++){
				// printf(" --> (%f, %f)", positions[i].x, positions[i].y);
				pos[i] = {int(positions[i].x+offset.x), int(positions[i].y+offset.y)};
				if(pos[i].x < 0 || pos[i].y < 0){
					continue;
				}
				cv::circle(bg_copy, pos[i], 3, (0, 0, 255), 3);
				cv::putText(bg_copy, std::to_string(int(poseIds[i])), cv::Point(int(pos[i].x)+10,int(pos[i].y)+10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,200), 1.5, CV_AA);
			}
			// std::cout << poseKeypoints.getSize(0)  << std::endl;
			if (poseKeypoints.getSize(0) * (poseKeypoints.getSize(0)-1) > 0){ // 距離リストを作成
				// 距離リストのメモリ割り当てでsegmentation fault が発生したのでここのデータ管理構造は要検討
				// float distance_list[3][(poseKeypoints.getSize(0) * (poseKeypoints.getSize(0)-1)/2)];
				// std::cout << sizeof(distance_list)/sizeof(*distance_list)  << std::endl;
				int list_index = 0;
				for(int i = 0; i < poseKeypoints.getSize(0)-1; i++){
					for(int j = i+1; j < poseKeypoints.getSize(0); j++){
						float d = sqrtf( pow(positions[i].x - positions[j].x, 2) + pow(positions[i].y - positions[j].y, 2) );
						// distance_list[list_index] = {float(poseIds[i]), float(poseIds[j]), d};
						// std::cout << d << std::endl;
						// list_index++;
						if (d/area_resize_rate < 2.0){
							cv::line(bg_copy, pos[i], pos[j], cv::Scalar(0,0,255), 2, CV_AA);
							// printf("2m Alert: (%d, %d)\n", int(poseIds[i]), int(poseIds[j]));
						}
					}
				}
			}
			// 描画処理
			cv::imshow("World_Tracking", bg_copy);
			writer2 << bg_copy;
            // op::log("People IDs: " + datumsPtr->at(0)->poseIds.toString(), op::Priority::High);
            // op::log("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);
            // op::log("Face keypoints: " + datumsPtr->at(0)->faceKeypoints.toString(), op::Priority::High);
            // op::log("Left hand keypoints: " + datumsPtr->at(0)->handKeypoints[0].toString(), op::Priority::High);
            // op::log("Right hand keypoints: " + datumsPtr->at(0)->handKeypoints[1].toString(), op::Priority::High);
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
        const auto key = (char)cv::waitKey(1);
        return (key == 27);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return true;
    }
}

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
        if (info.pos == 4){
        	std::cout << "Please press any key." << std::endl;
			std::vector<cv::Point> poly;
			for (int i = 0; i < 4; i++) {
				poly.push_back(cv::Point(info.Pt[i]));
			}
			cv::polylines(info.img, poly, true, cv::Scalar::all(255), 3, cv::LINE_AA);
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

cv::Mat getM (cv::Mat frame, int resize_rate_x, int resize_rate_y)//変換行列の作成
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

	// ４つの対応点
	// cv::Point2f srcPoint[4] = { { 0, 0 },{ 2000, 0 },{ 2000, 2000 },{ 0, 2000 } };
	cv::Point2f srcPoint[4] = info.Pt;
	cv::Point2f dstPoint[4] = { { 0, 0 },{float(resize_rate_x), 0 },{float(resize_rate_x), float(resize_rate_y) },{ 0, float(resize_rate_y) } };
	cv::Mat M = cv::getPerspectiveTransform(srcPoint,dstPoint);
	// cv::Mat M = cv::getPerspectiveTransform(dstPoint,srcPoint);
	// 確認
	// cv::imshow("test",cpy_frame);
	// cv::waitKey(0);
	return M;
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
        const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
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


//         // == ORIGINAL CODE OPENPOSE.CPP == //
//         // Configure OpenPose
//         op::log("Configuring OpenPose...", op::Priority::High);
//         op::Wrapper opWrapper;
//         configureWrapper(opWrapper);

//         // Start, run, and stop processing - exec() blocks this thread until OpenPose wrapper has finished
//         op::log("Starting thread(s)...", op::Priority::High);
//         opWrapper.exec();

//         // Measuring total time
//         op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);
//         // == ORIGINAL CODE OPENPOSE.CPP == //

        // Configuring OpenPose
        op::log("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        configureWrapper(opWrapper);

        // Starting OpenPose
        op::log("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        // // Read frames on directory
        // const auto imagePaths = op::getFilesOnDirectory(FLAGS_image_dir, op::Extensions::Images);

        // test用フィールドの作成
        int area_resize_rate = 50;
		// int mark_area_x = int(area_resize_rate * 8.23);
		// int mark_area_y = int(area_resize_rate * 5.485);
		int mark_area_x = int(area_resize_rate * 2.0);
		int mark_area_y = int(area_resize_rate * 2.0);
		int offset_x = int(area_resize_rate * 4.0);
		int offset_y = int(area_resize_rate * 4.0);
		int field_x = mark_area_x + 2*offset_x;
		int field_y = mark_area_y + 2*offset_y;
		cv::Mat bg = cv::Mat::zeros(field_y, field_x , CV_8UC3);
    	int cols = bg.cols;
    	int rows = bg.rows;
	    for (int j = 0; j < rows; j++) {
	        for (int i = 0; i < cols; i++) {
            bg.at<cv::Vec3b>(j, i)[0] = 161; //青
            bg.at<cv::Vec3b>(j, i)[1] = 195; //緑
            bg.at<cv::Vec3b>(j, i)[2] = 204; //赤
	        }
	    }
		// cv::line(bg, cv::Point(offset_x, offset_y), cv::Point(x, h-1), (255, 0, 0));
		cv::Point offset = {offset_x, offset_y};
		cv::rectangle(bg, offset, cv::Point(offset_x+mark_area_x, offset_y+mark_area_y), cv::Scalar(0,0,0), 2, 2);

        cv::VideoCapture cap(FLAGS_video);
        if(!cap.isOpened())//カメラデバイスが正常にオープンしたか確認．
        {
        //読み込みに失敗したときの処理
        return -1;
        }
        cv::Mat frame; //取得したフレーム
        if(cap.read(frame))
        	if (frame.cols > 1500){
        		cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
        	}
        	cv::Mat M = getM(frame, mark_area_x, mark_area_y);
        	std::cout << M << std::endl;

		// Define the codec and create VideoWriter object
		int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
		cv::VideoWriter writer1, writer2;
		int fps = 30;
		std::string outname = FLAGS_video.substr(1+FLAGS_video.find_last_of("/"), FLAGS_video.find_last_of(".")-1-FLAGS_video.find_last_of("/"));
		// std::cout << outname << std::endl;
		writer1.open("/data/output/" + outname + "output1.avi", fourcc, fps, cv::Size(frame.cols, frame.rows));
		writer2.open("/data/output/" + outname + "output2.avi", fourcc, fps, cv::Size(field_x, field_y));

        op::log("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();
        while(cap.read(frame))//無限ループ
        {
        	if (frame.cols > 1500){
        		cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
        	}
            const auto imageToProcess = frame;

        // // Process and display images
        // for (const auto& imagePath : imagePaths)
        // {
            // const auto imageToProcess = cv::imread(imagePath);
            auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
            if (datumProcessed != nullptr)
            {
                // printKeypoints(datumProcessed, bg, M, offset);

                const auto userWantsToExit = printKeypoints(datumProcessed, 
                	bg, M, offset, area_resize_rate,
                	writer1, writer2);
                if (userWantsToExit)
                {
                    op::log("User pressed Esc to exit demo.", op::Priority::High);
                    break;
                }
            }
            else
                // op::log("Image " + imagePath + " could not be processed.", op::Priority::High);
                op::log("Video could not be processed.", op::Priority::High);
        }

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
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
