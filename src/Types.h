//
// Created by pan_jinquan@163.com on 2020/6/24.
//


#ifndef BODY_DETECTION_RK3399_TYPES_H
#define BODY_DETECTION_RK3399_TYPES_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "debug.h"

using namespace std;

#define PI  3.141592653589793
//双线性差值INTER_LINEAR,速度较慢，效果较好；最邻近插值INTER_NEAREST，速度较快，效果较差
//static int INTER_FLAGS = cv::INTER_LINEAR;
static int INTER_FLAGS = cv::INTER_NEAREST;


/***
 * 模型基本参数bias和scale与torch的transforms数据处理的对齐方法
 * transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
 * torch : y = (x-m)/std=x/std-m/std; 由于torch中0~255归一化0~1,逆归一化后：
 *         y = x/std/255 - m/std
 * TNN   : y = scale*x + bias
 * 对比得 ：scale=1/std/255 , bias=-m/std
 */
struct ModelParam {
    float aspect_ratio;                //长宽比，一般为0.75
    float scale_ratio;                 //缩放比例，一般为1.25
    int input_width;                   //模型输入宽度，单位：像素
    int input_height;                  //模型输入高度，单位：像素
    bool use_udp;                      //是否使用无偏估计UDP,一般为false
    bool use_rgb;                      //是否使用RGB作为模型输入
    vector<float> bias;                //输入数据偏置：bias=-m/std
    vector<float> scale;               //输入数据归一化尺度：scale=1/std/255
    vector<vector<int>> skeleton;    //关键点连接序号ID(用于可视化显示)
};



/***
 * person Model param
 */
static ModelParam COCO_PERSON_PARAM = {0.75f,
                                      1.25f,
                                      192,
                                      256,
                                      false,
                                      false,
                                      {-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5, 0},//bias=-m/std
                                      {1 / 0.5 / 255.f, 1 / 0.5 / 255.f, 1 / 0.5 / 255.f, 0},     //scale=1/std/255
                                      {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6},
                                       {5, 7}, {6, 8}, {7, 9}, {8, 10}, {0, 1}, {0, 2}, {1, 3}, {2, 4}}};



static ModelParam MODEL_TYPE[] = {
        COCO_PERSON_PARAM,
};


/***
 * 关键点(包含一个坐标点point和分数score)
 */
struct KeyPoint {
    float score;//关键点分数
    cv::Point2f point;
};

/***
 * 目标信息(包含目标的多个关键点keypoints和检测区域框rect)
 */
struct ObjectInfo {
    vector<KeyPoint> keypoints;
    cv::Rect rect;
};

/***
 * 帧信息(帧图像中多个目标的信息)
 */
struct FrameInfo {
    vector<ObjectInfo> info;
};


#endif //BODY_DETECTION_RK3399_TYPES_H
