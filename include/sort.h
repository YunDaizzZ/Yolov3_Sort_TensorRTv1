#ifndef __SORT_H_
#define __SORT_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "kalmanfilter.h"
#include "hungarian.h"

using namespace std;
using namespace cv;

#define IOU_THRESHOLD 0.3
#define IOU_THRESHOLD2 0.7  // 增加新track时降低误检的影响，但是取多少合适呢。。。

struct DET_MSG {
	vector<vector<int>> now;
	vector<vector<int>> minus1;
	vector<vector<int>> minus2;
};

class KalmanBoxTracker {

    public:
        Kalman_Filter kf;
        int time_since_update;
        int id;
        vector<vector<float>> history;
        int hits;
        int hit_streak;
        int age;

    public:
        KalmanBoxTracker();
        ~KalmanBoxTracker() {
        }

        void initialize_variable(vector<int> bbox, int count);
        void update(vector<int> bbox);
        vector<float> predict();
        vector<float> get_state();

};

class Sort {

    public:
        int max_age;
        int min_hits;
        vector<KalmanBoxTracker> trackers;
        vector<float> types;
        int frame_count;

        DET_MSG aaa;  // 存储三帧探测信息
        vector<int> bbb;  // 记录三帧目标个数

    public:
        Sort();
        ~Sort() {
        }
        void initialize_variable(int age, int hits);
        vector<vector<int>> update(vector<vector<int>> dets, int& count);

};

float bbox_iou(vector<int> bbox1, vector<float> bbox2);
float bbox_iou(vector<int> bbox1, vector<int> bbox2);
vector<vector<float>> convert_bbox_to_z(vector<int> bbox);
vector<float> convert_x_to_bbox(vector<vector<float>> x);
void associate_detections_to_trackers(vector<vector<int>> dets, vector<vector<float>> trks, vector<vector<int>>& matched, 
                                    vector<int>& unmatched_dets, vector<int>& unmatched_trks);

#endif