// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"

#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <vector>

#include <math.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <string>
#include <iostream>

using namespace std;

float char_score_thresh = 0.3;
struct Object
{
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
    string text;
};

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net squeezenet;

    squeezenet.opt.use_vulkan_compute = true;

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models
    squeezenet.load_param("squeezenet_v1.1.param");
    squeezenet.load_model("squeezenet_v1.1.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("pool10", out);

    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}



std::string cnn_deocde(const ncnn::Mat score , std::vector<std::string> alphabetChinese,int n_class)
{
    float *srcdata = (float* ) score.data;
    string texts="";
//    std::vector<std::string> str_res;
//    std::vector<float> str_res_score;
    int *pred_index=new int[score.h];
    for (int i = 0; i < score.h;i++){
        int max_index = 0;

        float max_value = -1000;
        for (int j =0; j< score.w; j++){
            if (srcdata[ i * score.w + j ] > max_value){
                max_value = srcdata[i * score.w + j ];
                max_index = j;
            }
        }

        pred_index[i] = max_index;
        bool flag1 = (max_index != n_class-1);
        bool flag2 = (i>0) && (max_index == pred_index[i-1]);
        bool flag3 = (i>1) && (max_index == pred_index[i-2]);
        bool flag4 = max_value > char_score_thresh;
        if(flag1 && (!flag2 || flag3) && flag4 )
        {
//            str_res.push_back(alphabetChinese[max_index+1]);
//            str_res_score.push_back(max_value);
            texts += alphabetChinese[max_index+1];
        }
    }

    return texts;
}


static int detect_densenet(const cv::Mat& bgr)
{
    ncnn::Net densenet;

#if NCNN_VULKAN
    squeezenet.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    // the ncnn model https://github.com/nihui/ncnn-assets/tree/master/models

    char *param = "densenet-opt.param";
    char *bin = "densenet-opt.bin";
    char *keys_file = "cn_5990.txt";

    densenet.load_param(param);
    densenet.load_model(bin);

    //load keys
    std::vector<std::string>   alphabetChinese;
    int nclass = 5990;
    ifstream keys(keys_file);
    std::string filename;
    std::string line;

    if(keys) // 有该文件
    {
        while (getline (keys, line)) // line中不包括每行的换行符
        {
            alphabetChinese.push_back(line);
        }
        alphabetChinese.push_back("卍");
    }
    else // 没有该文件
    {
        std::cout <<"no txt file" << std::endl;
    }


    float k = bgr.rows/32.0;
    int target_w = int(bgr.cols/k);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_GRAY,
                                                 bgr.cols, bgr.rows, target_w, 32);
//    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_GRAY, bgr.cols, bgr.rows);

//    cv::Mat cv_img = cv::Mat::zeros(in.h,in.w,CV_8U);
//    in.to_pixels(cv_img.data,ncnn::Mat::PIXEL_GRAY);
//    imwrite("testbak.jpg", cv_img);

    const float mean_vals[1] = { 127.5};
    const float norm_vals[1] = { 1.0 /255.0};
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = densenet.create_extractor();
    ex.input("the_input", in);
    ncnn::Mat out;
    ex.extract("flatten/Reshape_2:0", out);

    ncnn::Mat cnn_preds;
    // batch fc
    ncnn::Mat flatten_Reshape_2(nclass, out.h);
    int num_thread = 4;
    for (int j = 0; j < out.h; j++)
    {
        ncnn::Extractor cnn_ex_1 = densenet.create_extractor();
        cnn_ex_1.set_num_threads(num_thread);
        ncnn::Mat blob_flatten_i = out.row_range(j, 1);
        cnn_ex_1.input("flatten/Reshape_2:0", blob_flatten_i);
        ncnn::Mat blob_out_i;
        cnn_ex_1.extract("out", blob_out_i);

        memcpy(flatten_Reshape_2.row(j), blob_out_i, nclass * sizeof(float));
    }

    cnn_preds = flatten_Reshape_2;
    string res_pre = cnn_deocde(cnn_preds,alphabetChinese,nclass);
    std::cout <<"-----------" <<std::endl;
    std::cout <<res_pre <<std::endl;
    return 0;
}


static int yolov3_densenet(const cv::Mat& bgr)
{
    ncnn::Net yolo3;
    ncnn::Net densenet;

    char *yolo3_param = "yolov3_text_15_opt.param";
    char *yolo3_bin = "yolov3_text_15_opt.bin";

    char *densenet_param = "densenet-opt.param";
    char *densenet_bin = "densenet-opt.bin";
    char *keys_file = "cn_5990.txt";

    yolo3.load_param(yolo3_param);
    yolo3.load_model(yolo3_bin);

    densenet.load_param(densenet_param);
    densenet.load_model(densenet_bin);

    //load keys
    std::vector<std::string>   alphabetChinese;
    int nclass = 5990;
    ifstream keys(keys_file);
    std::string filename;
    std::string line;

    if(keys) // 有该文件
    {
        while (getline (keys, line)) // line中不包括每行的换行符
        {
            alphabetChinese.push_back(line);
        }
        alphabetChinese.push_back("卍");
    }
    else // 没有该文件
    {
        std::cout <<"no txt file" << std::endl;
    }

    const float mean_vals[3] = {0.0f, 0.0f, 0.0f};
    const float norm_vals[3] = {1.0/255,1.0/255,1.0/255};
    const float mean_vals_cnn[1] = { 127.5};
    const float norm_vals_cnn[1] = { 1.0 /255.0};

    //yolo3
    int resize_w = 608;
    int resize_h = 608;
    ncnn::Mat in_ncnn = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR,
                                                      bgr.cols, bgr.rows, resize_w, resize_h);

    ncnn::Mat in_ncnn_src = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR,
                                                   bgr.cols, bgr.rows);

    in_ncnn.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex_yolo3 = yolo3.create_extractor();
    ex_yolo3.input("data", in_ncnn);
    ncnn::Mat out_yolo3;
    ex_yolo3.extract("output", out_yolo3);

    std::vector<Object> objects;
    int width = bgr.cols;
    int height = bgr.rows;
    for (int i=0; i<out_yolo3.h; i++)
    {
        const float* values = out_yolo3.row(i);
        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.x = values[2] * width;
        object.y = values[3] * height;
        object.w = values[4] * width - object.x;
        object.h = values[5] * height - object.y;
//        objects.push_back(object);

//      densenet识别网络
        float k = object.h/32.0;
        int target_w = int(object.w/k);

        unsigned char *pixels ;
        pixels= new unsigned char[height*width*3];
        in_ncnn_src.to_pixels(pixels, ncnn::Mat::PIXEL_BGR);

        const unsigned char* roi_data = (const unsigned char*)pixels + ((int)object.y * width + (int)object.x)*3 ;
        ncnn::Mat ncnn_roi = ncnn::Mat::from_pixels_resize(roi_data, ncnn::Mat::PIXEL_BGR2GRAY,
                                                           (int)object.w, (int)object.h, width*3, target_w, 32);

//        cv::Mat test0 = cv::Mat::zeros(ncnn_roi.h,ncnn_roi.w,CV_8UC1);
//        ncnn_roi.to_pixels(test0.data,ncnn::Mat::PIXEL_GRAY);
//        imwrite("test0.jpg", test0);

        ncnn_roi.substract_mean_normalize(mean_vals_cnn, norm_vals_cnn);
        ncnn::Extractor ex_cnn = densenet.create_extractor();
        ex_cnn.input("the_input", ncnn_roi);
        ncnn::Mat out_cnn;
        ex_cnn.extract("flatten/Reshape_2:0", out_cnn);


        // batch fc
        ncnn::Mat cnn_preds;
        ncnn::Mat flatten_Reshape_2(nclass, out_cnn.h);
        int num_thread = 4;
        for (int j = 0; j < out_cnn.h; j++)
        {
            ncnn::Extractor cnn_ex_1 = densenet.create_extractor();
            cnn_ex_1.set_num_threads(num_thread);
            ncnn::Mat blob_flatten_i = out_cnn.row_range(j, 1);
            cnn_ex_1.input("flatten/Reshape_2:0", blob_flatten_i);
            ncnn::Mat blob_out_i;
            cnn_ex_1.extract("out", blob_out_i);

            memcpy(flatten_Reshape_2.row(j), blob_out_i, nclass * sizeof(float));
        }

        cnn_preds = flatten_Reshape_2;
        string res_pre = cnn_deocde(cnn_preds,alphabetChinese,nclass);

        object.text = res_pre;
        objects.push_back(object);

        std::cout <<"-----------" <<std::endl;
        std::cout << res_pre <<std::endl;
    }

    return 0;
}



static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int main(int argc, char** argv)
{

//    detect_squeezenet
//    const char* imagepath = "00000004.jpg";
//    cv::Mat m = cv::imread(imagepath, cv::IMREAD_COLOR);

//    detect_densenet
//    const char* imagepath = "crop_resize_2.jpg";
//    cv::Mat m = cv::imread(imagepath, cv::IMREAD_GRAYSCALE);

//    yolov3_densenet
    const char* imagepath = "1095434.jpg";
    cv::Mat m = cv::imread(imagepath, cv::IMREAD_COLOR);

    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

//    std::vector<float> cls_scores;
//    detect_squeezenet(m, cls_scores);
//    print_topk(cls_scores, 3);


//    detect_densenet(m);
    yolov3_densenet(m);


    return 0;
}
