//
// Created by pan_jinquan@163.com on 2020/6/24.
//


#ifndef BODY_DETECTION_RK3399_BODY_DETECTION_H
#define BODY_DETECTION_RK3399_BODY_DETECTION_H

#pragma once

#include <vector>
#include <tnn/utils/blob_converter.h>
#include <tnn/core/tnn.h>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "Types.h"
#include "image_utils.h"

using namespace std;

#ifdef  TNN_ARM_ENABLE
static TNN_NS::DeviceType DEVICE_CPU = TNN_NS::DEVICE_ARM;
#else
static TNN_NS::DeviceType DEVICE_CPU = TNN_NS::DEVICE_NAIVE;
#endif

#ifdef  TNN_OPENCL_ENABLE
static TNN_NS::DeviceType DEVICE_GPU = TNN_NS::DEVICE_OPENCL;
#else
static TNN_NS::DeviceType DEVICE_GPU = TNN_NS::DEVICE_CUDA;
#endif

/***
 * TNN设备
 */
typedef enum {
    // run on CPU
    TNNCPU = 0,
    // run on GPU, if failed run on CPU(需要OpenCL支持)
    TNNGPU = 1,
    // run on NPU, if failed run on CPU(暂不支持NPU)
    TNNNPU = 2,
} TNNDevice;

/***
 * 符号函数
 * @param x
 * @return
 */
static int sign(float x) {
    if (x > 0) {
        return 1;
    } else if (x == 0) {
        return 0;
    } else {
        return -1;
    }
};


namespace dm {
    namespace vision {
        /***
         * 图像变换(transform)数据结构体
         */
        struct TransInfo {
            cv::Point2f center;       // 中心点
            cv::Point2f scale;        // 缩放比例
            cv::Rect rect;            // 检测框
            cv::Mat input_image;      // 根据rect裁剪的区域(检测区域)
        };

        class KeyPointDetector {
        public:
            /***
             * 构造函数
             * @param modelPath： TNN *.tnnmodel *.tnnproto文件路径（不含后缀名）
             * @param modelParam：模型输入参数
             * @param numThread: 开启线程数,默认1
             * @param TNNDevice: 运行设备，默认TNNCPU
             */
            KeyPointDetector(const string modelPath,
                             ModelParam modelParam,
                             int numThread=1,
                             TNNDevice deviceID=TNNCPU);


            /***
             * 构造函数
             * @param modelPath： TNN *.tnnmodel参数文件路径（含后缀名）
             * @param protoPath： TNN *.tnnproto模型文件路径（含后缀名）
             * @param modelParam：模型输入参数
             * @param numThread: 开启线程数,默认1
             * @param TNNDevice: 运行设备，默认TNNCPU
             */
            KeyPointDetector(const string modelPath,
                             const string protoPath,
                             ModelParam modelParam,
                             int numThread=1,
                             TNNDevice deviceID=TNNCPU);

            /***
             * 释放
             */
            ~KeyPointDetector();

            /***
             * 检测指尖关键点
             * @param imgBRG: BGR Image
             * @param boxes: 检测区域框(w,y,width,height)
             * @param scoreThresh：得分阈值，高于得分的坐标为有效值，低于该分数的坐标置为-1，值越大则结果越准确，范围：0~1，默认值：0.6
             * @param outFrameInfo: 输出当前帧图像的关键点信息
             * @return
             */
            int detect(cv::Mat &imgBRG, vector<cv::Rect> boxes,float scoreThresh, FrameInfo &outFrameInfo);


            /***
             * 可视化检测结果
             * @param imgBRG: BGR Image
             * @param outFrameInfo: 输出当前帧图像的关键点信息
             * @param waitKey: 等待key的时间，默认0
             */
            void visualize_result(cv::Mat &imgBRG,
                                  FrameInfo &frameInfo,
                                  vector<vector<int>> skeleton,
                                  int waitKey = 0);


        private:

            /***
             * 模型初始化
             * @param modelPath： TNN *.tnnmodel参数文件路径（含后缀名）
             * @param protoPath： TNN *.tnnproto模型文件路径（含后缀名）
             * @param TNNDevice: 运行设备，默认TNNCPU
             */
            int init_model(const string &modelPath, const string &protoPath, TNNDevice deviceID);

            /***
             * 预处理函数
             * @param image
             * @param rect
             * @param outTransInfo 输出图像变换(transform)数据结构体
             */
            void pre_process(cv::Mat &image, cv::Rect rect, TransInfo &outTransInfo);


            /***
             * 模型推理
             * @param input_image 模型输入的图像
             * @param outHeatmap  模型输出Tensor,即Heatmap
             * @return
             */
            int forward(cv::Mat &input_image, shared_ptr<TNN_NS::Mat> &outHeatmap);


            /***
             * 后处理函数
             * @param outObjectInfo 输出目标关键点信息
             * @param transInfo 图像变换(transform)数据结构体
             * @param heatmap
             * @param scoreThresh 关键点分数
             */
            void post_process(ObjectInfo &outObjectInfo,
                              TransInfo &transInfo,
                              shared_ptr<TNN_NS::Mat> &heatmap,
                              float scoreThresh = 0.5);

            /***
             *
             * @param transInfo 图像变换(transform)数据结构体
             * @param heatmap
             * @param coords
             * @param maxvals
             */
            void get_final_preds(TransInfo &transInfo,
                                 shared_ptr<TNN_NS::Mat> &heatmap,
                                 vector<cv::Point2f> &coords,
                                 vector<float> &maxvals);


            /***
             *
             * @param transInfo 图像变换(transform)数据结构体
             * @param heatmap
             * @param coords
             * @param maxvals
             */
            void get_final_preds_offset(TransInfo &transInfo,
                                        shared_ptr<TNN_NS::Mat> &heatmap,
                                        vector<cv::Point2f> &coords,
                                        vector<float> &maxvals);


            /***
             *
             * @param center
             * @param scale
             * @param rot
             * @param output_size
             * @param shift
             * @param inv
             * @return
             */
            cv::Mat get_affine_transform(cv::Point2f center,
                                         cv::Point2f scale,
                                         float rot,
                                         cv::Point2f output_size,
                                         cv::Point2f shift,
                                         bool inv = false);

            /***
             *
             * @param pt
             * @param t
             */
            void affine_transform(cv::Point2f &pt, cv::Mat t);


            /***
             *
             * @param pt
             * @param center
             * @param scale
             * @param output_size
             */
            void affine_transform_offset(cv::Point2f &pt,
                                         cv::Point2f center,
                                         cv::Point2f scale,
                                         cv::Point2f output_size);


            /***
             *
             * @param heatmap
             * @param preds
             * @param maxvals
             */
            void get_max_preds(shared_ptr<TNN_NS::Mat> &heatmap,
                               vector<cv::Point2f> &preds,
                               vector<float> &maxvals);


            /***
             *
             * @param heatmap
             * @param preds
             * @param maxvals
             */
            void get_max_preds_offset(shared_ptr<TNN_NS::Mat> &heatmap,
                                      vector<cv::Point2f> &preds,
                                      vector<float> &maxvals);


            /***
             *
             * @param a
             * @param b
             * @param o
             */
            void get_3rd_point(cv::Point2f a, cv::Point2f, cv::Point2f &o);

        public:
            float time_total{0.0f};           // 模型检测总共时间
            float time_pre_process{0.0f};     // 模型预处理时间
            float time_model_infer{0.0f};     // 模型推理时间
            float time_post_process{0.0f};    // 模型后处理时间
        private:
            const string TAG = "KeyPointDetectorCpp";
            shared_ptr<TNN_NS::TNN> net = nullptr;
            shared_ptr<TNN_NS::Instance> instance = nullptr;
            /***TNN_NS::DeviceType
             * DEVICE_NAIVE      = 0x0000,
             * DEVICE_X86        = 0x0010,
             * DEVICE_ARM        = 0x0020,
             * DEVICE_OPENCL     = 0x1000,
             * DEVICE_METAL      = 0x1010,
             * DEVICE_CUDA       = 0x1020,
             * DEVICE_DSP        = 0x1030,
             * DEVICE_ATLAS      = 0x1040,
             * DEVICE_HUAWEI_NPU = 0x1050,
             * DEVICE_RK_NPU     = 0x1060,
             */
            TNN_NS::DeviceType mDevice;  // 运行实例设备
            int mNumThread;
            ModelParam mModelParam;
        };
    }
};


#endif //BODY_DETECTION_RK3399_BODY_DETECTION_H
