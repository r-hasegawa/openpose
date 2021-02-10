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

static int DPID = 0;
static int DPS = 0;
static double TOTALDPTIME = 0.0;

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

struct DensePoints { // 密な場所
    std::vector<cv::Point2f> Pts;   // 密な場所の中心座標
    // int people = 0;    // dense point 内の人数
    std::vector<int> ids;
    std::vector<double> t; // 経過時間
    std::vector<int> sframe; // 開始フレーム
};

struct PotentialHumanPoints { // 人が隠れている可能性がある範囲
    std::vector<cv::Point2f> Pts;   // 人が隠れている可能性がある範囲の中心
    std::vector<int> ids;    // 隠れた人のID
    std::vector<int> ids_continuance; // 連続出現フレーム数 正の数なら連続検出 負の数なら連続未検出
};

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

bool printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr, 
    cv::Mat bg, cv::Mat evalimage, cv::Mat M, cv::Point offset, int area_resize_rate,
    cv::VideoWriter writer1, cv::VideoWriter writer2, int fnum,
    void *data1, void *data2,
    double t, double t2,
    std::ofstream &ofs, std::ofstream &ofs2, std::ofstream &ofs3)
{
    // std::cout << fnum << std::endl;
    // 1.Nose 2.Chest 3.RShoulder 4.RElbow 5.RWrist
    // 6.LShoulder 7.LElbow 8.LWrist 9.MidHip 10.RHip
    // 11.RKnee 12.RAnkle 13.LHip 14.LKnee 15.LAnkle
    // 16. REye 17.LEye 18.REar 19.LEar 20.Neck 21.Head
    const int key_id = 9;
    const int continuance_threshold = 1; // 連続検出フレーム数のしきい値


    DensePoints &dp = *(DensePoints *)data1;
    PotentialHumanPoints &php = *(PotentialHumanPoints *)data2;

    try
    {
        // Example: How to use the pose keypoints
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            cv::Mat disp_image = datumsPtr->at(0)->cvOutputData;
            auto poseIds = datumsPtr->at(0)->poseIds;
            auto poseKeypoints = datumsPtr->at(0)->poseKeypoints;

            float d;
            
            std::vector<cv::Point2f> points(poseKeypoints.getSize(0) + php.Pts.size());
            std::vector<cv::Point2f> positions(poseKeypoints.getSize(0) + php.Pts.size());
            std::vector<int> ids(poseIds.getSize(0) + php.Pts.size(), -1);
            std::vector<int> ids_frames(poseIds.getSize(0) + php.Pts.size(), 0);
            std::vector<int> ids_border(poseIds.getSize(0), -1);
            // Preprocessing
            int pos_num = 0;
            int php_num = 0;
            int delete_ids = 0;

            for(int i = 0; i < poseKeypoints.getSize(0); i++){
                const float Keypoint = poseKeypoints[i*21*3+key_id*3-1]; 
                if (Keypoint < 0.5
                    || poseKeypoints[i*21*3+(2)*3-1] < 0.5 // 2.Chestが見えるかどうか
                    ){
                    // std::cout << "Key point 9(MidHip) not detected" << std::endl;
                    continue;
                }
                int ids_index = vector_finder(php.ids, poseIds[i]);
                if (ids_index>=0){
                    if(php.ids_continuance[ids_index]>=0){// 連続で現れた場合
                        ids_frames[pos_num] = php.ids_continuance[ids_index]+1;
                    }
                    else{// 再び現れた場合
                        ids_frames[pos_num] = 1;
                    }
                }
                else{
                    // 初めて現れた場合
                    ids_frames[pos_num] = 1;
                }
                // std::cout << 19 * disp_image.cols/20 << std::endl;
                // std::cout << float(poseKeypoints[i*21*3+key_id*3-3]) << std::endl;
                float threshold_border = 20.0;
                cv::rectangle(disp_image, cv::Point(disp_image.cols/threshold_border,disp_image.rows/threshold_border)
                    , cv::Point((threshold_border-1.0)*disp_image.cols/threshold_border,(threshold_border-1.0)*disp_image.rows/threshold_border), cv::Scalar(0,255,0), 1, CV_AA);
                if (float(poseKeypoints[i*21*3+key_id*3-3]) < disp_image.cols/threshold_border
                    || float(poseKeypoints[i*21*3+key_id*3-3]) > (threshold_border-1.0) * disp_image.cols/threshold_border
                    || float(poseKeypoints[i*21*3+key_id*3-2]) < disp_image.rows/threshold_border
                    || float(poseKeypoints[i*21*3+key_id*3-2]) > (threshold_border-1.0) * disp_image.rows/threshold_border
                    ){ //画面外ギリギリ
                    ids_border[pos_num] = 1;
                }
                // std::cout << pos_num+php_num << std::endl;
                points[pos_num] = cv::Point2f(float(poseKeypoints[i*21*3+key_id*3-3]), float(poseKeypoints[i*21*3+key_id*3-2]) );
                ids[pos_num] = poseIds[i];
                cv::putText(disp_image, std::to_string(ids[pos_num]), cv::Point(int(points[pos_num].x + 20),int(points[pos_num].y + 15)), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,255,255), 3, CV_AA);
                


                // 腰の位置を調整
                // 足の角度が X°　以上 or 足の比率が長い方の0.75倍
                float d_leg_left = -1;
                if(float(poseKeypoints[i*21*3+(10)*3-1]) > 0.05
                 && float(poseKeypoints[i*21*3+(11)*3-1]) > 0.05 
                 && float(poseKeypoints[i*21*3+(12)*3-1]) > 0.05){
                    d_leg_left = std::max(d_leg_left, 2*std::max(
                        sqrtf( pow(float(poseKeypoints[i*21*3+(11)*3-3]) - float(poseKeypoints[i*21*3+(10)*3-3]), 2) 
                        + pow(float(poseKeypoints[i*21*3+(11)*3-2]) - float(poseKeypoints[i*21*3+(10)*3-2]), 2) ),
                        sqrtf( pow(float(poseKeypoints[i*21*3+(12)*3-3]) - float(poseKeypoints[i*21*3+(11)*3-3]), 2) 
                        + pow(float(poseKeypoints[i*21*3+(12)*3-2]) - float(poseKeypoints[i*21*3+(11)*3-2]), 2) ))
                        - std::abs(float(poseKeypoints[i*21*3+(12)*3-2]) - float(poseKeypoints[i*21*3+(10)*3-2])));

                    // std::cout << sqrtf( pow(float(poseKeypoints[i*21*3+(11)*3-3]) - float(poseKeypoints[i*21*3+(10)*3-3]), 2) 
                    //     + pow(float(poseKeypoints[i*21*3+(11)*3-2]) - float(poseKeypoints[i*21*3+(10)*3-2]), 2) ) << std::endl;
                    // std::cout << std::abs(float(poseKeypoints[i*21*3+(12)*3-2]) - float(poseKeypoints[i*21*3+(10)*3-2])) << std::endl;
                    // std::cout << d_leg << std::endl;
                    // std::cout << 2*std::max(
                    //     sqrtf( pow(float(poseKeypoints[i*21*3+(11)*3-3]) - float(poseKeypoints[i*21*3+(10)*3-3]), 2) 
                    //     + pow(float(poseKeypoints[i*21*3+(11)*3-2]) - float(poseKeypoints[i*21*3+(10)*3-2]), 2) ),
                    //     sqrtf( pow(float(poseKeypoints[i*21*3+(12)*3-3]) - float(poseKeypoints[i*21*3+(11)*3-3]), 2) 
                    //     + pow(float(poseKeypoints[i*21*3+(12)*3-2]) - float(poseKeypoints[i*21*3+(11)*3-2]), 2) ))
                    //     - std::abs(float(poseKeypoints[i*21*3+(12)*3-2]) - float(poseKeypoints[i*21*3+(10)*3-2])) << std::endl;
                    // std::cout << d_leg << std::endl;
                    // float theta = std::atan2((float(poseKeypoints[i*21*3+(11)*3-2]) - float(poseKeypoints[i*21*3+(10)*3-2]))
                    //     / (float(poseKeypoints[i*21*3+(11)*3-3]) - float(poseKeypoints[i*21*3+(10)*3-3])));
                }
                float d_leg_right = -1;
                if(float(poseKeypoints[i*21*3+(13)*3-1]) > 0.05
                 && float(poseKeypoints[i*21*3+(14)*3-1]) > 0.05 
                 && float(poseKeypoints[i*21*3+(15)*3-1]) > 0.05){
                    d_leg_right = std::max(d_leg_right, 2*std::max(
                        sqrtf( pow(float(poseKeypoints[i*21*3+(14)*3-3]) - float(poseKeypoints[i*21*3+(13)*3-3]), 2) 
                        + pow(float(poseKeypoints[i*21*3+(14)*3-2]) - float(poseKeypoints[i*21*3+(13)*3-2]), 2) ),
                        sqrtf( pow(float(poseKeypoints[i*21*3+(15)*3-3]) - float(poseKeypoints[i*21*3+(14)*3-3]), 2) 
                        + pow(float(poseKeypoints[i*21*3+(15)*3-2]) - float(poseKeypoints[i*21*3+(14)*3-2]), 2) ))
                        - std::abs(float(poseKeypoints[i*21*3+(15)*3-2]) - float(poseKeypoints[i*21*3+(13)*3-2])));
                }
                // std::cout << d_leg << std::endl;
                if(d_leg_left > 0 && d_leg_right){
                    points[pos_num].y = points[pos_num].y - std::min(d_leg_right,d_leg_left);
                    cv::circle(disp_image, points[pos_num], 6, cv::Scalar(255, 255, 255), -1);

                }

                pos_num++;
                // std::cout << pos_num << std::endl;
            }


            // ワールド座標系へ変換
            if ( poseKeypoints.getSize(0) > 0){ 
                cv::perspectiveTransform(points, positions, M);
            }

            // // 隠れた人の場所と検出された場所を比較
            // 1. IDの一致
            // 2. 近い順に結びつける
            // 3. 一定の距離以上離れており，結びつかなかった場合に追跡消失点にするかどうか判定
            // 3-1. 近くに人がいる(オクルージョンが起こり得る) or 直前のフレームにて検出されていた場合　追跡消失点とする
            // 3-2. 上記以外の場合 連続消失カウントをインクリメントし，　連続消失カウントが10を超えたときその追跡消失点を消滅させる

            if (php.Pts.size() > 0 && fnum > 0) {
                std::vector<std::vector<float>> d_php2ps(php.Pts.size()*(pos_num), std::vector<float>(3, 1000));
                std::vector<int> is_tracked_php(php.Pts.size(), -1); // 
                std::vector<int> is_tracked_pos(pos_num, -1); // 
                for(int j = 0; j < php.Pts.size(); j++){
                    if(vector_finder(ids, php.ids[j])>=0){// 追跡できた場合のphpを削除
                        // if(sqrtf( pow(php.Pts[j].x - positions[vector_finder(ids, php.ids[j])].x, 2) + pow(php.Pts[j].y - positions[vector_finder(ids, php.ids[j])].y, 2) ) / area_resize_rate < 5.0){
                        if(true){
                            is_tracked_php[j] = 1;
                            is_tracked_pos[vector_finder(ids, php.ids[j])] = 1;
                            continue;
                        }
                    }
                    for (int k = 0; k < pos_num; k++){ // php と現在の検出位置との距離リスト作成
                        d_php2ps[j*pos_num+k][0]  = sqrtf( pow(php.Pts[j].x - positions[k].x, 2) + pow(php.Pts[j].y - positions[k].y, 2) ) / area_resize_rate;
                        d_php2ps[j*pos_num+k][1] = j;
                        d_php2ps[j*pos_num+k][2] = k;
                    }
                }
                // 距離順にソート
                // std::cout << d_php2ps << std::endl;
                sort(d_php2ps.begin(),d_php2ps.end(),[](const std::vector<float> &alpha,const std::vector<float> &beta){return alpha[0] < beta[0];});
                // std::cout << d_php2ps << std::endl;
                // 距離順に結びつけ
                if(true){
                // if(d_php2ps[0][0] <= 4.0){
                    // std::cout << d_php2ps << std::endl;
                    std::vector<int> is_near_php(php.Pts.size(), -1); // 
                    for(int j = 0; j < d_php2ps.size(); j++){
                        if(d_php2ps[j][0] > 4.0){
                            break;
                        }
                        if(d_php2ps[j][0] <= 2.0 && is_near_php[d_php2ps[j][1]] == -1){
                            is_near_php[d_php2ps[j][1]] = 1;
                        }
                        if(is_tracked_php[d_php2ps[j][1]] == -1 && is_tracked_pos[d_php2ps[j][2]] == -1){
                            is_tracked_php[d_php2ps[j][1]] = 2;
                            is_tracked_pos[d_php2ps[j][2]] = 2;
                            // positions[pos_num+php_num] = php.Pts[j];
                            // ids[pos_num+php_num] = php.ids[j];
                            // php_num++;
                            // std::cout << fnum << " Swapped ID " << php.ids[d_php2ps[j][1]] << " => " << ids[d_php2ps[j][2]] << std::endl;
                        }
                    }
                    // std::cout << is_tracked_php << std::endl;
                    // std::cout << is_near_php << std::endl;
                    // std::cout << is_tracked_pos << std::endl;
                    // std::cout << ids_border << std::endl;
                    // std::cout << ids_frames << std::endl;
                    // std::cout << php.ids_continuance << std::endl;
                    for(int j = 0; j < is_tracked_php.size(); j++){
                        if(is_tracked_php[j] == -1){
                        // if(is_tracked_php[j] == -1 && is_near_php[j] > 0 && ids_border[j] < 0){ // 追跡できていないphp ==> 近くに他のIDがいる場合残す
                            if(php.ids_continuance[j] > 0 || is_near_php[j] > 0){
                                positions[pos_num+php_num] = php.Pts[j];
                                ids[pos_num+php_num] = php.ids[j];                           
                                ids_frames[pos_num+php_num] = -1;
                                php_num++; 
                                // std::cout << fnum << " New Occulusion ID " << php.ids[j] << std::endl;
                            }
                            else{
                                if(php.ids_continuance[j] > -10 ){
                                positions[pos_num+php_num] = php.Pts[j];
                                ids[pos_num+php_num] = php.ids[j];
                                ids_frames[pos_num+php_num] = php.ids_continuance[pos_num+php_num]-1;
                                php_num++; 
                                
                                }
                                else{
                                    // 10フレーム連続で検出できなかった場合，検出消失点を削除
                                    // std::cout << fnum << " Delete Occulusion ID " << php.ids[j] << std::endl;
                                }
                            }
                        }
                        // else{ // すでにトラッキング済み = ID一致
                        //     if(is_tracked_php[j] == -1){
                        //     std::cout << "Delete Occulusion ID " << php.ids[j] << std::endl;
                        //     }
                        // }
                    }
                    // for(int j = 0; j < pos_num; j++){
                    //     if(is_tracked_pos[j] < 0){ // 追跡できていないpos(新規ID) ==> 画面端の場合以外削除(1フレーム目を除く)
                    //         if(ids_border[j] < 0){
                    //             // 削除
                    //             std::cout << "Delete ID " << ids[j] << std::endl;
                    //             ids[j] = -ids[j];
                    //             delete_ids++;
                    //         }
                    //         else{
                    //             // 外部から来た新規ID
                    //             std::cout << "New ID " << ids[j] << std::endl;
                    //         }
                    //     }
                    // }
                }
            }
            

            int Endpoint = pos_num+php_num-delete_ids;      

            // 描画処理
            cv::Mat bg_copy = bg.clone();
            // cv::putText(bg_copy, std::to_string(fnum), cv::Point(int(bg_copy.cols)-100,int(bg_copy.rows)-10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 1.2, CV_AA);
            // cv::putText(bg_copy, "FPS:" + std::to_string(int(1000.0 / t)), cv::Point(int(bg_copy.cols)-200,30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 1.2, CV_AA);
            int s = int(t2/1000.0);
            int ms = int(t2/10.0) - s * 100;
            cv::putText(bg_copy, ("Time:" + std::to_string(s) + "." + std::to_string(ms)), cv::Point(30,30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 1.2, CV_AA);

            // このフレームで検出したすべて + 隠れていたもの
            std::vector<cv::Point2f> positions2(Endpoint);
            std::vector<int> ids2(Endpoint);
            // phpとして残すもの　+ 隠れていたもの
            std::vector<cv::Point2f> positions3(Endpoint);
            std::vector<int> ids3(Endpoint);

            std::vector<int> ids_frames2(Endpoint);

            int next_php_num = 0;

            for(int i = 0; i < Endpoint+delete_ids; i++){
                if(ids[i] < 0){
                    continue;
                }
                // positions2[i] = positions[i];
                // ids2[i] = ids[i];
                // if(positions2[i].x+offset.x < 0 || positions2[i].y+offset.y < 0){ // 範囲外
                //  continue;
                // }
                if(i < pos_num){ // 検出された人
                    if(ids_frames[i] >= continuance_threshold){ // 連続で検出されたフレーム数が xxx 以上のとき白丸候補
                        ids_frames2[next_php_num] = ids_frames[i];
                        ids2[next_php_num] = ids[i];
                        positions2[next_php_num] = positions[i];
                        ids3[next_php_num] = ids[i];
                        positions3[next_php_num] = positions[i];
                        next_php_num++;
                    }
                    cv::circle(bg_copy, cv::Point(int(positions2[i].x+offset.x), int(positions2[i].y+offset.y)), area_resize_rate*0.3, cv::Scalar(255, 0, 0), -1); // 直径30cmの円
                    cv::putText(bg_copy, std::to_string(ids2[i]), cv::Point(int(positions2[i].x+offset.x+10), int(positions2[i].y+offset.y+10)), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,0,0), 2, CV_AA);
                }else{ // 隠れた人
                    ids_frames2[next_php_num] = ids_frames[i];
                    ids2[next_php_num] = ids[i];
                    positions2[next_php_num] = positions[i];
                    ids3[next_php_num] = ids[i];
                    positions3[next_php_num] = positions[i];
                    next_php_num++;
                    // std::cout << pos[i] << std::endl;
                    cv::circle(bg_copy, cv::Point(int(positions2[i].x+offset.x), int(positions2[i].y+offset.y)), area_resize_rate*1.5/2, cv::Scalar(255, 255, 255), 2);
                    cv::putText(bg_copy, std::to_string(ids2[i]), cv::Point(int(positions2[i].x+offset.x+10), int(positions2[i].y+offset.y+10)), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2, CV_AA);
                }
            }
            // std::cout << ids << std::endl;
            // std::cout << ids2 << std::endl;
            // std::cout << php.Pts << std::endl;
            // std::cout << positions2 << std::endl;

            // 距離判定
            int num_dp = 0;
            std::vector<cv::Point2f> dps(positions2.size()*(positions2.size()-1));
            if (positions2.size() >= 2){ // 距離リストを作成
                for(int i = 0; i < positions2.size()-1; i++){
                    for(int j = i+1; j < positions2.size(); j++){
                        d = sqrtf( pow(positions2[i].x - positions2[j].x, 2) + pow(positions2[i].y - positions2[j].y, 2) );
                        if (d/area_resize_rate < 2.0){
                            cv::line(bg_copy, cv::Point(int(positions2[i].x+offset.x), int(positions2[i].y+offset.y)), 
                                cv::Point(int(positions2[j].x+offset.x), int(positions2[j].y+offset.y)), cv::Scalar(0,0,255), 2, CV_AA);
                            dps[num_dp] = (cv::Point2f(float((positions[i].x + positions[j].x)/2),float((positions[i].y + positions[j].y)/2)));
                            num_dp++;
                            // printf("2m Alert: (%d, %d)\n", int(poseIds[i]), int(poseIds[j]));
                        }
                    }
                }
            }

            // std::cout << dps << std::endl;

            // 密な場所のトラッキングと経過時間計測 ver1 
            float maxt;
            int trackid;
            std::vector<cv::Point2f> dps2(num_dp);
            std::vector<int> dpids(num_dp);
            std::vector<double> time(num_dp);
            std::vector<int> sf(num_dp);
            std::vector<int> status(dp.Pts.size(),1); // 1:引き継ぎ前 -1:引き継ぎ済
            ofs << fnum << ',' << num_dp;
            if ( num_dp > 0){
                for(int i = 0; i < num_dp; i++){
                    dps2[i] = dps[i];
                    maxt = 0.0;
                    trackid = -1;
                    for(int j = 0; j < dp.Pts.size(); j++){
                        if (status[j] < 0){
                            continue;
                        } 
                        d = sqrtf( pow(dps2[i].x - dp.Pts[j].x, 2) + pow(dps2[i].y - dp.Pts[j].y, 2) )/area_resize_rate;
                        if(d < 2.0){
                            // std::cout << dp.ids[j] << std::endl;
                            // std::cout << dp.t[j] << std::endl;
                            if (dp.t[j] > maxt) {
                                maxt = dp.t[j];
                                trackid = j;
                            }
                        }
                    }
                    // 距離2m以内の密な場所　最も長時間なものと結びつける
                    if (trackid >= 0){ // idの引き継ぎと時間の追加
                        status[trackid] = -1;
                        dpids[i] = dp.ids[trackid];
                        time[i] = dp.t[trackid] + t;
                        sf[i] = dp.sframe[trackid];
                    }else{
                        // std::cout << t << std::endl;
                        dpids[i] = DPID;
                        DPID++;
                        time[i] = t; // 初期値0.1秒
                        sf[i] = fnum;
                    }

                    cv::Scalar dense_color;
                    if (time[i]/1000 > 4.0){
                        dense_color = {0, 0, 255};
                    }else if (time[i]/1000 > 2.0) {
                        dense_color = {0, (255 * (2.0 - (time[i]/1000)/2.0)), 255};
                    }else{
                        dense_color = {0, 255, (255 * (time[i]/1000)/2.0)};
                    }
                    cv::circle(bg_copy, cv::Point(int(dps2[i].x+offset.x), int(dps2[i].y+offset.y)), 
                            area_resize_rate*2.0/2, dense_color, 2);
                    cv::putText(bg_copy, std::to_string(int(time[i]/1000)), cv::Point(int(dps2[i].x+offset.x), int(dps2[i].y+offset.y)), 
                        cv::FONT_HERSHEY_SIMPLEX, 1, dense_color, 1.5, CV_AA);
                    ofs << ',' << dpids[i] << ',' << dps2[i].x/area_resize_rate << ',' << dps2[i].y/area_resize_rate << ',' << time[i]/1000;
                    ofs3 << fnum << ',' << dpids[i] << ',' << dps2[i].x/area_resize_rate << ',' << dps2[i].y/area_resize_rate << ',' << time[i]/1000 << std::endl;
                }
            }
            ofs << std::endl;

            for(int j = 0; j < dp.Pts.size(); j++){ // 削除された密ID
                if (status[j] < 0){
                    continue;
                } 
                // std::cout << fnum << " Delete Dense ID " << dp.ids[j] << std::endl;
                TOTALDPTIME = TOTALDPTIME + dp.t[j]/1000;
                cv::Scalar dense_color;

                if (dp.t[j]/1000 > 4.0){
                    dense_color = {0, 0, 255};
                    DPS = DPS + 1;
                }else if (dp.t[j]/1000 > 2.0) {
                    dense_color = {0, (255 * (2.0 - (dp.t[j]/1000)/2.0)), 255};
                }else{
                    dense_color = {0, 255, (255 * (dp.t[j]/1000)/2.0)};
                }
                cv::circle(evalimage, cv::Point(int(dp.Pts[j].x+offset.x), int(dp.Pts[j].y+offset.y)),
                            area_resize_rate*2.0/2, dense_color, 2);
                cv::putText(evalimage, std::to_string(int(dp.t[j]/1000)), cv::Point(int(dp.Pts[j].x+offset.x), int(dp.Pts[j].y+offset.y)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, dense_color, 1.5, CV_AA);
                ofs2 << dp.ids[j] << ','  << dp.sframe[j] << ',' << fnum-1  << ',' << dp.t[j]/1000 << ','  << dp.Pts[j].x/area_resize_rate << ','  << dp.Pts[j].y/area_resize_rate << ','  << std::endl;
            }
            cv::Mat evalimage_copy = evalimage.clone();
            cv::putText(evalimage_copy, "Close Pts:" + std::to_string(int(DPID)), cv::Point(int(bg_copy.cols)-300,30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 1.2, CV_AA);
            cv::putText(evalimage_copy, "(Dense Pts:" + std::to_string(int(DPS)) + ")", cv::Point(int(bg_copy.cols)-300,70), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 1.2, CV_AA);
            cv::putText(evalimage_copy, ("TOTAL Time:" + std::to_string(int(TOTALDPTIME)) + "." + std::to_string(int(TOTALDPTIME*100)-int(TOTALDPTIME)*100)), cv::Point(30,30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 1.2, CV_AA);



            //
            // phpに今回の検出結果を追加
            php.Pts = positions3; // 隠れたID & 検出できたID
            php.ids = ids3; // 隠れたID & 検出できたID
            php.ids_continuance = ids_frames2; // 検出したIDについての連続で出現した回数
            dp.Pts = dps2;
            dp.ids = dpids;
            dp.t = time;
            dp.sframe = sf;

            // printf("Past:");
            // for (int i = 0; i < php.Pts.size(); i++){
            //     printf("%d, ", php.ids[i]);
            // }
            // printf("\n");
            // printf("Now :");
            // for (int i = 0; i < ids2.size(); i++){
            //     printf("%d, ", ids2[i]);
            // }
            // printf("\n");
            
            // 出力
            cv::putText(disp_image, std::to_string(fnum), cv::Point(int(disp_image.cols)-200,int(disp_image.rows)-20), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255,255,255), 2.4, CV_AA);
            
            // gousei
            cv::Mat bg_copy2;
            cv::resize(bg_copy, bg_copy2, cv::Size(), 1.5, 1.5);
            cv::Mat roi = disp_image(cv::Rect(0, 0, bg_copy2.cols, bg_copy2.rows));
            bg_copy2.copyTo(roi);

            // cv::Mat roi3 = disp_image(cv::Rect(bg_copy2.cols+20, 0, bg_copy3.cols, bg_copy3.rows));
            // bg_copy3.copyTo(roi3);

            writer1 << disp_image;
            writer2 << bg_copy;

			// 表示
			if (true){
                if (disp_image.cols > 1500){
                    cv::resize(disp_image, disp_image, cv::Size(), 0.5, 0.5);
                }
				cv::imshow("OpenPose Tracking", disp_image);
				// cv::imshow("Tracking", bg_copy);
    //             cv::resize(evalimage_copy, evalimage_copy, cv::Size(), 0.6, 0.6);
    //             cv::imshow("Total", evalimage_copy);
			}
            // op::log("People IDs: " + datumsPtr->at(0)->poseIds.toString(), op::Priority::High);
            // op::log("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);
            // op::log("Face keypoints: " + datumsPtr->at(0)->faceKeypoints.toString(), op::Priority::High);
            // op::log("Left hand keypoints: " + datumsPtr->at(0)->handKeypoints[0].toString(), op::Priority::High);
            // op::log("Right hand keypoints: " + datumsPtr->at(0)->handKeypoints[1].toString(), op::Priority::High);

        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
        const auto key = (char)cv::waitKey(1);
        if (key == 32){
            php.Pts = {};
            php.ids = {};
        }
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
        std::cout << "info.Pt[" << info.pos << "] = {" << x << "," << y << "};"<< std::endl;
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
    if (false){
    	// コールバック関数を登録する
	    info.winName = "test";
	    cv::namedWindow(info.winName);
	    cv::setMouseCallback(info.winName, mouseCallback, (void *)&info);
	    cv::imshow(info.winName, info.img);
	    cv::waitKey();
	    cv::destroyAllWindows();
    }else{
		// ４つの対応点
		// cv::Point2f srcPoint[4] = { {左上 },{ 右上 },{ 右下 },{ 左下 } };
		// info.Pt[0] = { 997, 513 };
		// info.Pt[1] = { 1380, 540 };
		// info.Pt[2] = { 1489, 895 };
		// info.Pt[3] = { 1026, 898 };
        // std::string fname = FLAGS_video.compare(1 + FLAGS_video.find_last_of("/", FLAGS_video.find_last_of("/")-1), 2, "Bv");
        if(FLAGS_video.compare(1 + FLAGS_video.find_last_of("/", FLAGS_video.find_last_of("/")-1), 2, "Bv") == 0){
            // front camera
            std::cout << "front camera" << std::endl;
            // info.Pt[0] = {585,450}; 4x8 m
            // info.Pt[1] = {1503,474};
            // info.Pt[2] = {1703,813};
            // info.Pt[3] = {341,783};
            info.Pt[0] = {788,482};
            info.Pt[1] = {1260,495};
            info.Pt[2] = {1377,857};
            info.Pt[3] = {657,823};
        }else if(FLAGS_video.compare(1 + FLAGS_video.find_last_of("/", FLAGS_video.find_last_of("/")-1), 2, "Bh") == 0){
            std::cout << "side camera" << std::endl;
            info.Pt[0] = {675, 617};
            info.Pt[1] = {739, 425};
            info.Pt[2] = {1116, 421};
            info.Pt[3] = {1193, 611};
        }else{
            std::cout << "naname camera" << std::endl;
            info.Pt[0] = {706, 579};
            info.Pt[1] = {1000, 455};
            info.Pt[2] = {1342, 517};
            info.Pt[3] = {1104, 711}; 
        }


        // 3m 100cm
        // info.Pt[0] = { 512, 185 };
        // info.Pt[1] = { 1355, 177 };
        // info.Pt[2] = { 2955, 1142 };
        // info.Pt[3] = { -70, 1173 };
        // // 4.5m
        // info.Pt[0] = { 1004, 412 };
        // info.Pt[1] = { 1754, 556 };
        // info.Pt[2] = { 1148, 786 };
        // info.Pt[3] = { 365, 530 };
        // // 6m 90cm
        // info.Pt[0] = {523,79};
        // info.Pt[1] = {1351,81};
        // info.Pt[2] = {1719,700};
        // info.Pt[3] = {149,698};

    }
    cv::Point2f srcPoint[4] = info.Pt;
	cv::Point2f dstPoint[4] = { { 0, 0 },{float(resize_rate_x), 0 },{float(resize_rate_x), float(resize_rate_y) },{ 0, float(resize_rate_y) } };
	cv::Mat M = cv::getPerspectiveTransform(srcPoint,dstPoint);
	// cv::Mat M = cv::getPerspectiveTransform(dstPoint,srcPoint);
	// 確認
	// cv::imshow("test",cpy_frame);
	// cv::waitKey(0);
	return M;
}

std::vector<int> zoomImage (cv::Mat frame)//変換行列の作成
{
    // 射影変換のための画像情報構造体
    ImageInfo info;
    // 画像を読み込む
    info.img = frame;
    if (true){
        // コールバック関数を登録する
        info.winName = "zoom";
        cv::namedWindow(info.winName);
        cv::setMouseCallback(info.winName, mouseCallback, (void *)&info);
        cv::imshow(info.winName, info.img);
        cv::waitKey();
        cv::destroyAllWindows();
    }else{
        // ４つの対応点
        // cv::Point2f srcPoint[4] = { {左上 },{ 右上 },{ 右下 },{ 左下 } };
        info.Pt[0] = { 997, 513 };
        info.Pt[1] = { 1380, 540 };
    }
    std::vector<int> zoom(4);
    zoom[0] = 10;
    zoom[0] = int(info.Pt[0].x);
    zoom[1] = int(info.Pt[0].y);
    zoom[2] = int(info.Pt[1].x - info.Pt[0].x);
    zoom[3] = int(info.Pt[1].y - info.Pt[0].y);
    return zoom;
}

void draw_tennis_court(cv::Mat bg, cv::Point offset, cv::Point offset2, int arr, bool hor){
    cv::Point tl = {offset.x+offset2.x,offset.y+offset2.y};
    if (hor){
        cv::rectangle(bg, cv::Point(tl.x+int(arr*(-7.8)),tl.y+int(arr*(-5.5)))
            , cv::Point(tl.x+int(arr*(23.77 + 7.8)),tl.y+int(arr*(10.97 + 5.5))), cv::Scalar(129,219,193), -1, CV_AA);
        cv::rectangle(bg, tl, cv::Point(tl.x+int(arr*23.77),tl.y+int(arr*10.97)), cv::Scalar(233,160,0), -1, CV_AA);
        cv::rectangle(bg, tl, cv::Point(tl.x+int(arr*23.77),tl.y+int(arr*10.97)), cv::Scalar(255,255,255), 2, 2);
        cv::rectangle(bg, cv::Point(tl.x,tl.y+int(arr*1.37)), cv::Point(tl.x+int(arr*23.77),tl.y+int(arr*(10.97-1.37))), cv::Scalar(255,255,255), 2, 2);
        cv::rectangle(bg, cv::Point(tl.x+int(arr*5.485),tl.y+int(arr*1.37)), cv::Point(tl.x+int(arr*(23.77-5.485)),tl.y+int(arr*(10.97-1.37))), cv::Scalar(255,255,255), 2, 2);
        cv::line(bg, cv::Point(tl.x+int(arr*11.885),tl.y), cv::Point(tl.x+int(arr*11.885),tl.y+int(arr*(10.97))), cv::Scalar(255,255,255), 2, CV_AA);
        cv::line(bg, cv::Point(tl.x+int(arr*5.485),tl.y+int(arr*5.485)), cv::Point(tl.x+int(arr*(23.77-5.485)),tl.y+int(arr*(5.485))), cv::Scalar(255,255,255), 2, CV_AA);
    }else{
        cv::rectangle(bg, cv::Point(tl.x+int(arr*(-5.5)),tl.y+int(arr*(-7.8)))
            , cv::Point(tl.x+int(arr*(10.97 + 5.5)),tl.y+int(arr*(23.77 + 7.8))), cv::Scalar(129,219,193), -1, CV_AA);
        cv::rectangle(bg, tl, cv::Point(tl.x+int(arr*10.97),tl.y+int(arr*23.77)), cv::Scalar(233,160,0), -1, CV_AA);
        cv::rectangle(bg, tl, cv::Point(tl.x+int(arr*10.97),tl.y+int(arr*23.77)), cv::Scalar(255,255,255), 2, 2);
        cv::rectangle(bg, cv::Point(tl.x+int(arr*1.37),tl.y), cv::Point(tl.x+int(arr*(10.97-1.37)),tl.y+int(arr*23.77)), cv::Scalar(255,255,255), 2, 2);
        cv::rectangle(bg, cv::Point(tl.x+int(arr*1.37),tl.y+int(arr*5.485)), cv::Point(tl.x+int(arr*(10.97-1.37)),tl.y+int(arr*(23.77-5.485))), cv::Scalar(255,255,255), 2, 2);
        cv::line(bg, cv::Point(tl.x,tl.y+int(arr*11.885)), cv::Point(tl.x+int(arr*(10.97)),tl.y+int(arr*11.885)), cv::Scalar(255,255,255), 2, CV_AA);
        cv::line(bg, cv::Point(tl.x+int(arr*5.485),tl.y+int(arr*5.485)), cv::Point(tl.x+int(arr*(5.485)),tl.y+int(arr*(23.77-5.485))), cv::Scalar(255,255,255), 2, CV_AA);
    }
}

void draw_eval_court(cv::Mat bg, cv::Point offset, cv::Point offset2, int arr, int area_x, int area_y){
    cv::Point tl = {offset.x+offset2.x,offset.y+offset2.y};
    int i;
    for(i=0;i<=area_x;i++){
        cv::line(bg, cv::Point(tl.x+int(arr*i),tl.y), cv::Point(tl.x+int(arr*i),tl.y+int(arr*(area_y))), cv::Scalar(255,255,255), 2, CV_AA);
    }
    for(i=0;i<=area_y;i++){
        cv::line(bg, cv::Point(tl.x,tl.y+int(arr*i)), cv::Point(tl.x+int(arr*area_x),tl.y+int(arr*i)), cv::Scalar(255,255,255), 2, CV_AA);
    }
    
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

        // // Read frames on directory
        // const auto imagePaths = op::getFilesOnDirectory(FLAGS_image_dir, op::Extensions::Images);

        // test用フィールドの作成
        int area_resize_rate = 30;
		int mark_area_x = int(area_resize_rate * 4);
		int mark_area_y = int(area_resize_rate * 4);
		// int mark_area_x = int(area_resize_rate * 2.0);
		// int mark_area_y = int(area_resize_rate * 2.0);
		int offset_x = int(area_resize_rate * 3.0);
		int offset_y = int(area_resize_rate * 2.0);
		int field_x = mark_area_x + 1*offset_x + int(area_resize_rate * 3.0);
		int field_y = mark_area_y + 2*offset_y;
		cv::Mat bg = cv::Mat::zeros(field_y, field_x , CV_8UC3);
    	int cols = bg.cols;
    	int rows = bg.rows;
	    for (int j = 0; j < rows; j++) {
	        for (int i = 0; i < cols; i++) {
            bg.at<cv::Vec3b>(j, i)[0] = 68; //青
            bg.at<cv::Vec3b>(j, i)[1] = 153; //緑
            bg.at<cv::Vec3b>(j, i)[2] = 0; //赤
            // bg.at<cv::Vec3b>(j, i)[0] = 161; //青
            // bg.at<cv::Vec3b>(j, i)[1] = 215; //緑
            // bg.at<cv::Vec3b>(j, i)[2] = 252; //赤
	        }
	    }
		// cv::line(bg, cv::Point(offset_x, offset_y), cv::Point(x, h-1), (255, 0, 0));
		cv::Point offset = {offset_x, offset_y};
        // draw_tennis_court(bg, offset, cv::Point(int(area_resize_rate*-(0)),int(area_resize_rate*-(0))), area_resize_rate, false); // 背景画像　オフセット　オフセット2 resize 縦向き
        draw_eval_court(bg, offset, cv::Point(int(area_resize_rate*-(2.0)),int(area_resize_rate*-(0))), area_resize_rate, 8, 4); // 背景画像　オフセット　オフセット2 resize 縦向き
		// cv::rectangle(bg, offset, cv::Point(offset_x+mark_area_x, offset_y+mark_area_y), cv::Scalar(255,255,255), 2, 2);
        
        cv::Mat evalimage = bg.clone();
        cv::Mat frame; //取得したフレーム

        // imwrite("bg.png", bg);

        cv::VideoCapture cap(FLAGS_video);
        if(!cap.isOpened())//カメラデバイスが正常にオープンしたか確認．
        {
            //読み込みに失敗したときの処理
            return -1;
        }

        if (std::equal(FLAGS_video.begin(), FLAGS_video.end(), "/dev/video0")){
            op::log("Use Web Camera", op::Priority::High);
            cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920); // カメラ画像の横幅を1280に設定
            cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080); // カメラ画像の縦幅を720に設定
            for(int i = 0;i<10;i++){
                cap.read(frame);
            }
        }



        std::vector<int> zoom(4);

        cv::Rect rect;
        cv::Mat M;

        if(cap.read(frame)){
        if (frame.cols > 1920){
         cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
        }
        zoom[2] = frame.cols;
        zoom[3] = frame.rows;
        // zoom = zoomImage(frame);
        rect = cv::Rect(zoom[0], zoom[1], zoom[2], zoom[3]);
        M = getM(frame(rect), mark_area_x, mark_area_y);
        // std::cout << M << std::endl;
        }



		// Define the codec and create VideoWriter object
		int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
		cv::VideoWriter writer1, writer2;
		int fps = 30;
        int skip_frame = 1;
        fps = fps/skip_frame;
        std::string outname;
		outname = FLAGS_video.substr(1+FLAGS_video.find_last_of("/", FLAGS_video.find_last_of("/")-1), FLAGS_video.find_last_of(".")-1-FLAGS_video.find_last_of("/", FLAGS_video.find_last_of("/")-1));
		// std::cout << "Please Input Filename" << std::endl;
  //       std::cin >> outname;
        // std::cout << "OUTPUT NAME = " << outname << std::endl;
        // return 0;

        // std::cout << outname << std::endl;
		writer1.open("/data/eval_output/" + outname + "_output1.mp4", fourcc, fps, cv::Size(zoom[2], zoom[3]));
		writer2.open("/data/eval_output/" + outname + "_output2.mp4", fourcc, fps, cv::Size(field_x, field_y));

        std::ofstream ofs("/data/eval_csv/" + outname + "_1.csv");
        std::ofstream ofs2("/data/eval_csv/" + outname + "_2.csv");
        std::ofstream ofs3("/data/eval_csv/" + outname + "_3.csv");
        if (!ofs || !ofs2 || !ofs3){
            std::cout << "Could not open CSV file" << std::endl;
            return 0;
        }



        DensePoints dp;
        PotentialHumanPoints php;

        op::log("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();
        int fnum = 0;
        double FPS =fps;
        std::chrono::system_clock::time_point Lstms = std::chrono::system_clock::now();
        std::chrono::system_clock::time_point Lstms2;
        Lstms2 = Lstms;
        cap.set(CV_CAP_PROP_POS_FRAMES,0);
        while(cap.read(frame))//無限ループ
        {
            auto nowms = std::chrono::system_clock::now();
            auto diff = nowms - Lstms; // スタートからの経過時間
            auto diff2 = nowms - Lstms2; // １フレームごとの経過時間
            Lstms = nowms;
            // FPS = 1000.0 / double(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count());
            // std::cout << FPS << std::endl;



        	if (frame.cols > 1920){
        		cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
        	}
            const auto imageToProcess = frame(rect);;

        // // Process and display images
        // for (const auto& imagePath : imagePaths)
        // {
            // const auto imageToProcess = cv::imread(imagePath);
            auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
            if (datumProcessed != nullptr)
            {
                // printKeypoints(datumProcessed, bg, M, offset);

                // const auto userWantsToExit = printKeypoints(datumProcessed, 
                // 	bg, evalimage, M, offset, area_resize_rate,
                // 	writer1, writer2, fnum,
                //     (void *)&dp, (void *)&php,
                //     double(std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()), double(std::chrono::duration_cast<std::chrono::milliseconds>(diff2).count())
                //     ,ofs);
                const auto userWantsToExit = printKeypoints(datumProcessed, 
                    bg, evalimage, M, offset, area_resize_rate,
                    writer1, writer2, fnum,
                    (void *)&dp, (void *)&php,
                    double(1000/fps), double((1000/fps)*fnum),
                    ofs, ofs2, ofs3);
                if (userWantsToExit)
                {
                    op::log("User pressed Esc to exit demo.", op::Priority::High);
                    fnum++;
                    break;
                }
            }
            else
                // op::log("Image " + imagePath + " could not be processed.", op::Priority::High);
                op::log("Video could not be processed.", op::Priority::High);
        fnum++;
        }

        for(int j = 0; j < dp.Pts.size(); j++){
            ofs2 << dp.ids[j] << ','  << dp.sframe[j] << ','  << fnum  << ',' << dp.t[j]/1000 << ','  << dp.Pts[j].x/area_resize_rate << ','  << dp.Pts[j].y/area_resize_rate << ','  << std::endl;
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
