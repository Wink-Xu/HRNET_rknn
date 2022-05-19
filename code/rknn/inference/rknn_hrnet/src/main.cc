// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"


#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#include "rknn_api.h"

using namespace std;
#define POSE_KP_NUM 17
/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp))
    {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if (fp)
    {
        fclose(fp);
    }
    return model;
}


int heatmapToKeypoints(float* buffer, float* key_points)
{
    // heatmap to keypoints
	int kpts_num = POSE_KP_NUM;
	int i, j, k;
    int hm_width = 64, hm_height = 48;

	for(k = 0; k < kpts_num; ++k)
	{
		float* th_map = buffer + k * hm_width * hm_height;
		float conf = 0.0;
		int x = 0, y = 0;
		
		for(i = 0; i < hm_height; ++i)
		{
			const float* th_row = th_map + i * hm_width;
			for(j = 0; j < hm_width; ++j)
			{
				if(th_row[j] > conf)
				{
					x = j;
					y = i;
					conf = th_row[j];
				}
			}
		}
		key_points[3*k] = x * 4;
		key_points[3*k+1] = y * 4;
		key_points[3*k+2] = conf;
	}

    return 0;
}

static unsigned char *load_image(const char *image_path, rknn_tensor_attr *input_attr)
{
    int req_height = 0;
    int req_width = 0;
    int req_channel = 0;

    switch (input_attr->fmt)
    {
    case RKNN_TENSOR_NHWC:
        req_height = input_attr->dims[2];
        req_width = input_attr->dims[1];
        req_channel = input_attr->dims[0];
        break;
    case RKNN_TENSOR_NCHW:
        req_height = input_attr->dims[1];
        req_width = input_attr->dims[0];
        req_channel = input_attr->dims[2];
        break;
    default:
        printf("meet unsupported layout\n");
        return NULL;
    }

    printf("w=%d,h=%d,c=%d, fmt=%d\n", req_width, req_height, req_channel, input_attr->fmt);

    int height = 0;
    int width = 0;
    int channel = 0;

    unsigned char *image_data = stbi_load(image_path, &width, &height, &channel, req_channel);
    if (image_data == NULL)
    {
        printf("load image failed!\n");
        return NULL;
    }

    if (width != req_width || height != req_height)
    {
        unsigned char *image_resized = (unsigned char *)STBI_MALLOC(req_width * req_height * req_channel);
        if (!image_resized)
        {
            printf("malloc image failed!\n");
            STBI_FREE(image_data);
            return NULL;
        }
        if (stbir_resize_uint8(image_data, width, height, 0, image_resized, req_width, req_height, 0, channel) != 1)
        {
            printf("resize image failed!\n");
            STBI_FREE(image_data);
            return NULL;
        }
        STBI_FREE(image_data);
        image_data = image_resized;
    }

    return image_data;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{

#ifdef pTime
    struct  timeval tv_1, tv_2, tv_3, tv_4, tv_5, tv_6;
    gettimeofday(&tv_1,NULL);
#endif
    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;

    const char *model_path = argv[1];
    const char *img_path = argv[2];

    // Load RKNN Model
    model = load_model(model_path, &model_len);

    ret = rknn_init(&ctx, model, model_len, 0);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }
#ifdef pTime
    gettimeofday(&tv_2,NULL);
    printf("\n load_model = %.2f ms\n", (float)((tv_2.tv_sec - tv_1.tv_sec)*1000 + (tv_2.tv_usec - tv_1.tv_usec)*0.001));
#endif
    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs[i]));
    }

#ifndef pTime
    // Load image
    cv::Mat orig_img = imread(img_path, cv::IMREAD_COLOR);
    if(!orig_img.data) {
        printf("cv::imread %s fail!\n", img_path);
        return -1;
    }

    cv::Mat img = orig_img.clone();
    if(orig_img.cols != 256 || orig_img.rows != 192) {
        printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, 256, 192);
        cv::resize(orig_img, img, cv::Size(256, 192), (0, 0), (0, 0), cv::INTER_LINEAR);
    }
#endif

    unsigned char *input_data = NULL;
    input_data = load_image(img_path, &input_attrs[0]);
    if (!input_data)
    {
        return -1;
    }

#ifdef pTime
    gettimeofday(&tv_3,NULL);
    printf("\n load_image = %.2f ms\n", (float)((tv_3.tv_sec - tv_2.tv_sec)*1000 + (tv_3.tv_usec - tv_2.tv_usec)*0.001));
#endif


    // Set Input Data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = input_attrs[0].size;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = input_data;

    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

#ifdef pTime
    gettimeofday(&tv_4,NULL);
    printf("\n rknn_inputs_set = %.2f ms\n", (float)((tv_4.tv_sec - tv_3.tv_sec)*1000 + (tv_4.tv_usec - tv_3.tv_usec)*0.001));
#endif
    ret = rknn_run(ctx, nullptr);
#ifdef pTime
    gettimeofday(&tv_5,NULL);
    printf("\n rknn_run = %.2f ms\n", (float)((tv_5.tv_sec - tv_4.tv_sec)*1000 + (tv_5.tv_usec - tv_4.tv_usec)*0.001));
#endif
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }

    // Post Process
    float *buffer = (float *)outputs[0].buf;
    float *keypoints = (float*)malloc(17*3*sizeof(float));
    heatmapToKeypoints(buffer, keypoints);


#ifndef pTime
    printf("result: \n");
	cv::Point point;//特征点，用以画在图像中  
    for(int i =0; i < 17*3;)
    {
        printf("[%d,%d,%.2f]\n", (int)keypoints[i], (int)keypoints[i+1], keypoints[i+2]);
        i+= 3;
        point.x = (int)keypoints[i];
        point.y  = (int)keypoints[i+1];
        cv::circle(img, point, 4, cv::Scalar(0, 0, 255));//在图像中画出特征点，2是圆的半径 
    }
    cv::imwrite("./out.jpg", img);
#endif
    // Release rknn_outputs
    rknn_outputs_release(ctx, 1, outputs);

#ifdef pTime
    gettimeofday(&tv_6,NULL);
    printf("\n rknn_outputs_get and postprocess = %.2f ms\n", (float)((tv_6.tv_sec - tv_5.tv_sec)*1000 + (tv_6.tv_usec - tv_5.tv_usec)*0.001));
#endif
    // Release
    if (ctx >= 0)
    {
        rknn_destroy(ctx);
    }
    if (model)
    {
        free(model);
    }

    if (input_data)
    {
        stbi_image_free(input_data);
    }
    
    return 0;
}






