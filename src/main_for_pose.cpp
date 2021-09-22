//
// Created by pan_jinquan@163.com on 2020/6/24.
//


#include "KeyPointDetector.h"
#include "Types.h"
#include <iostream>
#include <string>
#include <vector>

using namespace dm;
using namespace vision;
using namespace std;



void test_coco_person() {
    // 人体姿态检测
    const char *test_image = "../data/test_image/person/test2.jpg";
    const char *model_path = (char *) "../data/tnn/coco_person/mbv2_1.0_g_17_192_256_128_s1.25_sim.opt";
    const int num_thread = 1;
    const float score_thresh = 0.5;

    TNNDevice device_id = TNNGPU;//运行设备
    ModelParam model_param = COCO_PERSON_PARAM;//模型参数
    KeyPointDetector *detector = new KeyPointDetector(model_path,
                                                      model_param,
                                                      num_thread,
                                                      device_id);
    cv::Mat bgr;
    bgr = cv::imread(test_image);
    int src_h = bgr.rows;
    int src_w = bgr.cols;
    // 检测区域为整张图片的大小
    FrameInfo resultInfo;
    cv::Rect box(0, 0, src_w, src_h);
    vector<cv::Rect> bboxes;
    bboxes.push_back(box);
    // 开始检测
    detector->detect(bgr, bboxes, score_thresh, resultInfo);
    // 可视化代码
    detector->visualize_result(bgr, resultInfo, model_param.skeleton);
    // 释放空间
    delete detector;
    detector = nullptr;
    printf("FINISHED.\n");
}


int main() {
    test_coco_person();
    return 0;
}
