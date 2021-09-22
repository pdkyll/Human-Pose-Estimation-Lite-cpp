//
// Created by pan_jinquan@163.com on 2020/6/24.
//

#include "KeyPointDetector.h"
#include "debug.h"
#include "file_utils.h"

namespace dm {
    namespace vision {
        KeyPointDetector::KeyPointDetector(const string modelPath,
                                           ModelParam modelParam,
                                           int numThread,
                                           TNNDevice deviceID) {
            this->mModelParam = modelParam;
            this->mNumThread = numThread;
            string tnnproto = modelPath + ".tnnproto";
            string tnnmodel = modelPath + ".tnnmodel";
            // Load param and model
            int status = init_model(tnnmodel, tnnproto, deviceID);
            if (status == -1) {
                LOGI("TNN init failed %d", status);
            } else {
                LOGI("TNN init successfully: %d", status);
            }
        }


        KeyPointDetector::KeyPointDetector(const string modelPath,
                                           const string protoPath,
                                           ModelParam modelParam,
                                           int numThread,
                                           TNNDevice deviceID) {
            this->mModelParam = modelParam;
            this->mNumThread = numThread;
            // Load param and model
            int status = init_model(modelPath, protoPath, deviceID);
            if (status == -1) {
                LOGI("TNN init failed %d", status);
            } else {
                LOGI("TNN init successfully: %d", status);
            }
        }

        int KeyPointDetector::init_model(const string &modelPath,
                                         const string &protoPath,
                                         TNNDevice deviceID) {
            //vector<int> nchw = {1, 3, this->mInputHeight, this->mInputWidth};
            vector<int> nchw = {1, 3, this->mModelParam.input_height,
                                this->mModelParam.input_width};
            string protoContent, modelContent;

            if (!file_exists(modelPath)) {
                LOGI("no tnnmodel file:%s", modelPath.c_str());
                return -1;

            }
            if (!file_exists(protoPath)) {
                LOGI("no tnnproto file:%s", protoPath.c_str());
                return -1;
            }
            LOGI("load tnnproto %s", protoPath.c_str());
            LOGI("load tnnmodel %s", modelPath.c_str());
            protoContent = load_file(protoPath);
            modelContent = load_file(modelPath);
            LOGI("tnnproto len=%d", protoContent.length());
            LOGI("tnnmodel len=%d", modelContent.length());
            this->mDevice = deviceID == TNNGPU ? DEVICE_GPU : DEVICE_CPU;
            TNN_NS::Status status;
            TNN_NS::ModelConfig config;

            config.model_type = TNN_NS::MODEL_TYPE_TNN;
            //config.model_type = TNN_NS::MODEL_TYPE_NCNN;
            config.params = {protoContent, modelContent};

            net = make_shared<TNN_NS::TNN>();
            status = net->Init(config);

            if (status != TNN_NS::TNN_OK) {
                LOGI("detector init failed %d", (int) status);
                return -1;
            }

            TNN_NS::InputShapesMap shapeMap;
            //shapeMap.insert(pair<string, TNN_NS::DimsVector>("input", nchw));
            //instance
            TNN_NS::NetworkConfig network_config;
            network_config.library_path = {""};
            network_config.device_type = this->mDevice;
            instance = net->CreateInst(network_config, status, shapeMap);
            //instance->SetCpuNumThreads(std::max(this->mNumThread, 1));
            if (this->mDevice == DEVICE_CPU) {
                // fix a BUG:Error Init layer Clip_131 (err: 40966 or 0xA006)
                instance->SetCpuNumThreads(std::max(this->mNumThread, 1));
            }
            if (status != TNN_NS::TNN_OK || !instance) {
                LOGI("DEVICE_GPU:%d initialization failed, switch to DEVICE_CPU", this->mDevice);
                // 如果出现GPU加载失败，切换到CPU
                this->mDevice = DEVICE_CPU;
                network_config.device_type = this->mDevice;
                instance = net->CreateInst(network_config, status, shapeMap);
                instance->SetCpuNumThreads(std::max(this->mNumThread, 1));
                if (status != TNN_NS::TNN_OK) {
                    LOGI("detector init failed %%lld", (int) status);
                    return -1;
                }
            }
            return status == TNN_NS::TNN_OK ? 0 : -1;
        }

        KeyPointDetector::~KeyPointDetector() {
            this->net->DeInit();
            this->instance->DeInit();
            this->net = nullptr;
            this->instance = nullptr;
        }

        int
        KeyPointDetector::detect(cv::Mat &imgBRG, vector<cv::Rect> boxes, float scoreThresh, FrameInfo &outFrameInfo) {
            this->time_pre_process = 0.f;
            this->time_model_infer = 0.f;
            this->time_post_process = 0.f;
            this->time_total = 0.f;

            if (imgBRG.empty()) {
                LOGI("image is empty ,please check!");
                return 0;
            }
            int nums = boxes.size();
            DEBUG_TIME(t0);
            for (int i = 0; i < nums; ++i) {
                DEBUG_TIME(t1);
                TransInfo transInfo;
                this->pre_process(imgBRG, boxes.at(i), transInfo);
                DEBUG_TIME(t2);
                shared_ptr<TNN_NS::Mat> heatmap;
                this->forward(transInfo.input_image, heatmap);
                DEBUG_TIME(t3);
                ObjectInfo outObjectInfo;
                this->post_process(outObjectInfo, transInfo, heatmap, scoreThresh);
                DEBUG_TIME(t4);
                outFrameInfo.info.push_back(outObjectInfo);
                this->time_pre_process += RUN_TIME(t2 - t1);
                this->time_model_infer += RUN_TIME(t3 - t2);
                this->time_post_process += RUN_TIME(t4 - t3);
            }
            DEBUG_TIME(t5);
            this->time_total = RUN_TIME(t5 - t0);
            LOGW("===================Benchmark========================");
            LOGW("-->pre_process  : %3.5f/%d=%3.5f ms", this->time_pre_process, nums, this->time_pre_process / nums);
            LOGW("-->model_infer  : %3.5f/%d=%3.5f ms", this->time_model_infer, nums, this->time_model_infer / nums);
            LOGW("-->post_process : %3.5f/%d=%3.5f ms", this->time_post_process, nums, this->time_post_process / nums);
            LOGW("-->avg_total    : %3.5f/%d=%3.5f ms", this->time_total, nums, this->time_total / nums);
            LOGW("====================================================");
            return 0;
        }


        void KeyPointDetector::pre_process(cv::Mat &image, cv::Rect rect, TransInfo &outTransInfo) {
            float x = rect.x;
            float y = rect.y;
            float w = rect.width;
            float h = rect.height;
            outTransInfo.rect = rect;
            //float aspect_ratio = 0.75;
            //float scale_ratio = 1.25;
            float pixel_std = 200;
            cv::Point2f center;
            cv::Point2f scale;
            center.x = x + w * 0.5;
            center.y = y + h * 0.5;
            if (w > this->mModelParam.aspect_ratio * h) {
                h = w * 1.0 / this->mModelParam.aspect_ratio;
            } else if (w < this->mModelParam.aspect_ratio * h) {
                w = h * this->mModelParam.aspect_ratio;
            }
            scale.x = w * 1.0 / pixel_std;
            scale.y = h * 1.0 / pixel_std;
            if (center.x != -1) {
                scale.x = scale.x * this->mModelParam.scale_ratio;
                scale.y = scale.y * this->mModelParam.scale_ratio;
            }
            cv::Point2f output_size(this->mModelParam.input_width, this->mModelParam.input_height);
            float rot = 0;
            cv::Mat trans = get_affine_transform(center,
                                                 scale,
                                                 rot,
                                                 output_size,
                                                 {0., 0.},
                                                 false);

            cv::warpAffine(image, outTransInfo.input_image, trans,
                           cv::Size(output_size.x, output_size.y),
                           INTER_FLAGS);
            if (this->mModelParam.use_rgb) {
                cv::cvtColor(outTransInfo.input_image, outTransInfo.input_image, cv::COLOR_BGR2RGB);
            }
            outTransInfo.center = center;
            outTransInfo.scale = scale;
            DEBUG_COUT(<< "trans = " << endl << " " << trans);
            DEBUG_PRINT("center: [%f,%f]", center.x, center.y);
            DEBUG_PRINT("scale : [%f,%f]", scale.x, scale.y);
        }

        cv::Mat KeyPointDetector::get_affine_transform(cv::Point2f center,
                                                       cv::Point2f scale,
                                                       float rot,
                                                       cv::Point2f output_size,
                                                       cv::Point2f shift,
                                                       bool inv) {

            float scale_tmp_0 = scale.x * 200.0;
            float scale_tmp_1 = scale.y * 200.0;
            float src_w = scale_tmp_0;
            float dst_w = (float) output_size.x;
            float dst_h = (float) output_size.y;
            float rot_rad = PI * rot / 180;
            float src_point0 = 0;
            float src_point1 = -0.5 * src_w;
            float sn = sin(rot_rad);
            float cs = cos(rot_rad);
            //src_dir = get_dir([0, src_w * -0.5], rot_rad)
            //dst_dir = np.array([0, dst_w * -0.5], np.float32)
            float src_dir_x = src_point0 * cs - src_point1 * sn;
            float src_dir_y = src_point0 * sn + src_point1 * cs;
            float dst_dir_x = 0;
            float dst_dir_y = -0.5 * dst_w;
            //src = np.zeros((3, 2), dtype = np.float32)
            //dst = np.zeros((3, 2), dtype = np.float32)
            cv::Mat src(3, 2, CV_32FC1);
            cv::Mat dst(3, 2, CV_32FC1);
            //src[0, :] = center + scale_tmp * shift
            src.at<float>(0, 0) = center.x + scale_tmp_0 * shift.x;
            src.at<float>(0, 1) = center.y + scale_tmp_1 * shift.y;
            //src[1, :] = center + src_dir + scale_tmp * shift
            src.at<float>(1, 0) = center.x + src_dir_x + scale_tmp_0 * shift.x;
            src.at<float>(1, 1) = center.y + src_dir_y + scale_tmp_1 * shift.y;
            //dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
            dst.at<float>(0, 0) = dst_w * 0.5;
            dst.at<float>(0, 1) = dst_h * 0.5;
            //dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) +dst_dir
            dst.at<float>(1, 0) = dst_w * 0.5 + dst_dir_x;
            dst.at<float>(1, 1) = dst_h * 0.5 + dst_dir_y;
            //src[2:, :] = get_3rd_point(src[0, :], src[1, :])
            //b + np.array([-direct[1], direct[0]], dtype=np.float32)
            cv::Point2f src_o = {0, 0};
            get_3rd_point({src.at<float>(0, 0), src.at<float>(0, 1)},
                          {src.at<float>(1, 0), src.at<float>(1, 1)}, src_o);
            cv::Point2f dst_o = {0, 0};
            get_3rd_point({dst.at<float>(0, 0), dst.at<float>(0, 1)},
                          {dst.at<float>(1, 0), dst.at<float>(1, 1)}, dst_o);
            src.at<float>(2, 0) = src_o.x;
            src.at<float>(2, 1) = src_o.y;
            dst.at<float>(2, 0) = dst_o.x;
            dst.at<float>(2, 1) = dst_o.y;
            cv::Mat trans;
            if (inv) {
                trans = cv::getAffineTransform(dst, src);
            } else {
                trans = cv::getAffineTransform(src, dst);
            }
            DEBUG_COUT(<< "src = " << endl << " " << src);
            DEBUG_COUT(<< "dst = " << endl << " " << dst);
            return trans;
        }

        void KeyPointDetector::get_3rd_point(cv::Point2f a, cv::Point2f b, cv::Point2f &o) {
            //direct = a - b
            //return b + np.array([-direct[1], direct[0]], dtype=np.float32)
            o.x = b.x - (a.y - b.y);
            o.y = b.y + (a.x - b.x);
        }


        int KeyPointDetector::forward(cv::Mat &input_image, shared_ptr<TNN_NS::Mat> &outHeatmap) {

            // 数据始终位于CPU，不需要设置成OPENCL，tnn自动复制cpu->gpu
            //TNN_NS::DimsVector target_dims = {1, 3, this->mInputHeight, this->mInputWidth};
            TNN_NS::DimsVector target_dims = {1, 3, this->mModelParam.input_height,
                                              this->mModelParam.input_width};
            //LOGW("instance device:%d", this->device);
            //LOGW("data device    :%d", DEVICE_CPU);
            auto input_tensor = make_shared<TNN_NS::Mat>(DEVICE_CPU,
                                                         TNN_NS::N8UC3, target_dims,
                                                         input_image.data);
            // step 1. set input mat
            TNN_NS::MatConvertParam input_convert_param;
            input_convert_param.scale = this->mModelParam.scale;
            input_convert_param.bias = this->mModelParam.bias;
            // TNN初始化时设置CPU或者GPU模式
            auto status = instance->SetInputMat(input_tensor, input_convert_param);
            if (status != TNN_NS::TNN_OK) {
                LOGE("SetInputMat Error: %s", status.description().c_str());
                return status;
            }

            // step 2. Forward
            status = instance->ForwardAsync(nullptr);
            //status = instance->Forward();
            if (status != TNN_NS::TNN_OK) {
                LOGE("Forward Error: %s", status.description().c_str());
                return status;
            }

            // step 3. get output mat
            TNN_NS::MatConvertParam output_convert_param;
            shared_ptr<TNN_NS::Mat> output_tensor = nullptr;
            status = instance->GetOutputMat(output_tensor, output_convert_param,
                                            "output",
                                            DEVICE_CPU,
                                            TNN_NS::NCHW_FLOAT);

            if (status != TNN_NS::TNN_OK) {
                LOGE("GetOutputMat Error: %s", status.description().c_str());
                return status;
            }

            outHeatmap = output_tensor;
            DEBUG_PRINT("input_tensor : w,h=%d,%d", input_tensor->GetWidth(),
                        input_tensor->GetHeight());
            DEBUG_PRINT("output_tensor: w,h=%d,%d", output_tensor->GetWidth(),
                        output_tensor->GetHeight());
            return status;
        }

        void KeyPointDetector::get_final_preds(TransInfo &transInfo,
                                               shared_ptr<TNN_NS::Mat> &heatmap,
                                               vector<cv::Point2f> &coords,
                                               vector<float> &maxvals) {

            get_max_preds(heatmap, coords, maxvals);
            int w = heatmap->GetWidth();
            int h = heatmap->GetHeight();
            int num_joints = heatmap->GetChannel(); //num joints
            auto *heatmaps_data = (float *) heatmap->GetData();// dim=(1,num_anchors,2,1) NCHW
            int wh = w * h;
            for (int c = 0; c < num_joints; ++c) {
                //const float *ptr = finger->heatmap->GetDims(c);
                const float *ptr = &heatmaps_data[c * wh];
                int x = int(floor(coords[c].x + 0.5));
                int y = int(floor(coords[c].y + 0.5));
                if (((1 < x) && (x < w - 1)) && ((1 < y) && (y < h - 1))) {
                    float diff_x = ptr[y * w + x + 1] - ptr[y * w + x - 1];
                    float diff_y = ptr[(y + 1) * w + x] - ptr[(y - 1) * w + x];
                    coords[c].x += sign(diff_x) * 0.25;
                    coords[c].y += sign(diff_y) * 0.25;
                }
            }
            float rot = 0;
            cv::Point2f output_size(w, h);
            cv::Mat trans = get_affine_transform(transInfo.center,
                                                 transInfo.scale,
                                                 rot,
                                                 output_size,
                                                 {0., 0.},
                                                 true);
            DEBUG_COUT(<< "inv-trans = " << endl << " " << trans);
            for (int c = 0; c < num_joints; ++c) {
                affine_transform(coords[c], trans);
            }
        }


        void KeyPointDetector::get_final_preds_offset(TransInfo &transInfo,
                                                      shared_ptr<TNN_NS::Mat> &heatmap,
                                                      vector<cv::Point2f> &coords,
                                                      vector<float> &maxvals) {
            int w = heatmap->GetWidth();
            int h = heatmap->GetHeight();
            int num_joints = heatmap->GetChannel() / 3; //num joints
            auto *heatmaps_data = (float *) heatmap->GetData();// dim=(1,num_anchors,2,1) NCHW
            int wh = w * h;
            float KPD = 4.0;
            get_max_preds_offset(heatmap, coords, maxvals);
            //DEBUG_COUT(<<"coords"<<coords);
            for (int c = 0; c < num_joints; ++c) {
                //const float *ptr = finger->heatmap->GetDims(c);
                //const float *ptr = &heatmaps_data[3 * c * wh];
                const float *offset_x = &heatmaps_data[(3 * c + 1) * wh];
                const float *offset_y = &heatmaps_data[(3 * c + 2) * wh];
                int x = int(coords[c].x);
                int y = int(coords[c].y);
                coords[c].x += offset_x[y * w + x] * KPD;
                coords[c].y += offset_y[y * w + x] * KPD;
            }
            cv::Point2f output_size(w, h);
            for (int c = 0; c < num_joints; ++c) {
                affine_transform_offset(coords[c], transInfo.center, transInfo.scale, output_size);
            }
        }


        void KeyPointDetector::post_process(ObjectInfo &outObjectInfo,
                                            TransInfo &transInfo,
                                            shared_ptr<TNN_NS::Mat> &heatmap,
                                            float scoreThresh) {
            vector<cv::Point2f> coords;
            vector<float> maxvals;
            if (this->mModelParam.use_udp) {
                get_final_preds_offset(transInfo, heatmap, coords, maxvals);
            } else {
                get_final_preds(transInfo, heatmap, coords, maxvals);
            }

            for (int i = 0; i < coords.size(); ++i) {
                KeyPoint kp;
                if (maxvals[i] > scoreThresh) {
                    kp.point = coords[i];
                } else {
                    kp.point = {-1., -1.};
                }
                kp.score = maxvals[i];
                outObjectInfo.keypoints.push_back(kp);
            }
            outObjectInfo.rect = transInfo.rect;
        }

        void KeyPointDetector::affine_transform(cv::Point2f &pt, cv::Mat t) {
            cv::Mat new_pt = (cv::Mat_<double>(3, 1) << pt.x, pt.y, 1.0);
            cv::Mat out = t * new_pt; // (2,3)*(3,1)=(2,1)
            pt.x = (float) out.at<double>(0, 0);
            pt.y = (float) out.at<double>(1, 0);
        }

        void KeyPointDetector::affine_transform_offset(cv::Point2f &pt,
                                                       cv::Point2f center,
                                                       cv::Point2f scale,
                                                       cv::Point2f output_size) {
            scale = scale * 200.f;
            float scale_x = scale.x / (output_size.x - 1.0);
            float scale_y = scale.y / (output_size.y - 1.0);
            pt.x = pt.x * scale_x + center.x - scale.x * 0.5;
            pt.y = pt.y * scale_y + center.y - scale.y * 0.5;
        }

        void KeyPointDetector::get_max_preds(shared_ptr<TNN_NS::Mat> &heatmap,
                                             vector<cv::Point2f> &preds,
                                             vector<float> &maxvals) {
            //int dims = batch_heatmaps->GetChannel();
            int w = heatmap->GetWidth();
            int h = heatmap->GetHeight();
            auto *heatmaps_data = (float *) heatmap->GetData();// dim=(1,num_anchors,2,1) NCHW
            int wh = w * h;
            int num_joints = heatmap->GetChannel(); //num joints
            for (int c = 0; c < num_joints; ++c) {
                //const float *ptr = batch_heatmaps.channel(c);
                const float *ptr = &heatmaps_data[c * wh];
                cv::Point2f point;
                float max_value = 0.0f;
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        if (ptr[x] >= max_value) {
                            point.x = x;
                            point.y = y;
                            max_value = ptr[x];
                        }
                    }
                    ptr += w;
                }
                preds.push_back(point);
                maxvals.push_back(max_value);
            }
        }


        void KeyPointDetector::get_max_preds_offset(shared_ptr<TNN_NS::Mat> &heatmap,
                                                    vector<cv::Point2f> &preds,
                                                    vector<float> &maxvals) {
            //int dims = batch_heatmaps->GetChannel();
            int w = heatmap->GetWidth();
            int h = heatmap->GetHeight();
            auto *heatmaps_data = (float *) heatmap->GetData();// dim=(1,num_anchors,2,1) NCHW
            int wh = w * h;
            int num_joints = heatmap->GetChannel() / 3; //num joints
            for (int c = 0; c < num_joints; ++c) {
                //const float *ptr = batch_heatmaps.channel(c);
                //const float *ptr = &heatmaps_data[c * wh];
                const float *ptr = &heatmaps_data[3 * c * wh];
                cv::Point2f point;
                float max_value = 0.0f;
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        if (ptr[x] >= max_value) {
                            point.x = x;
                            point.y = y;
                            max_value = ptr[x];
                        }
                    }
                    ptr += w;
                }
                preds.push_back(point);
                maxvals.push_back(max_value);
            }
        }


        void KeyPointDetector::visualize_result(cv::Mat &imgBRG,
                                                FrameInfo &frameInfo,
                                                vector<vector<int>> skeleton,
                                                int waitKey) {
            //draw rectangle
            cv::Mat vis_image;
            imgBRG.copyTo(vis_image);
            for (int i = 0; i < frameInfo.info.size(); ++i) {
                // draw rect
                auto obj = frameInfo.info.at(i);
                cv::Rect rect(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
                string obj_info = "ID:" + to_string(i);
                draw_rect_text(vis_image, rect, obj_info);
                // draw points
                vector<cv::Point2f> points;
                vector<string> texts;
                for (int j = 0; j < obj.keypoints.size(); ++j) {
                    auto kps = obj.keypoints[j];
                    LOGI("ID:%d point:[%f,%f] score:%f ", i, kps.point.x, kps.point.y, kps.score);
                    // string info = to_string(i) + " score:" + to_string(lm.score);
                    char info[200];
                    //sprintf(info, "%d-score:%3.3f", j, kps.score);
                    sprintf(info, "%d", j);
                    points.push_back(cv::Point(kps.point.x, kps.point.y));
                    texts.push_back(info);
                }
                draw_points_texts(vis_image, points, texts, cv::Scalar(0, 0, 255));
                draw_lines(vis_image, points, skeleton, cv::Scalar(255, 0, 0));
                //draw_arrowed_lines(vis_image, points, skeleton);
            }
            image_show("result", vis_image, waitKey);
            image_save("../result.jpg", vis_image);
        }

    }
};

