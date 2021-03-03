#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <opencv2/opencv.hpp>

#include "sort.h"

using namespace std;
using namespace cv;

float bbox_iou(vector<int> bbox1, vector<float> bbox2) {
    // bbox: [x1,y1,x2,y2]
    float interBox[] = {
        max(bbox1[0] / 1.f, bbox2[0]),
        max(bbox1[1] / 1.f, bbox2[1]),
        min(bbox1[2] / 1.f, bbox2[2]),
        min(bbox1[3] / 1.f, bbox2[3]),
    };

    if (interBox[1] >= interBox[3] || interBox[0] >= interBox[2])
        return 0.f;

    float interBoxS = (interBox[2] - interBox[0]) * (interBox[3] - interBox[1]);

    float b1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    float b2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);

    return interBoxS / (b1_area + b2_area - interBoxS + 1e-16);
}

float bbox_iou(vector<int> bbox1, vector<int> bbox2) {
    // bbox: [x1,y1,x2,y2]
    int interBox[] = {
        max(bbox1[0], bbox2[0]),
        max(bbox1[1], bbox2[1]),
        min(bbox1[2], bbox2[2]),
        min(bbox1[3], bbox2[3]),
    };

    if (interBox[1] >= interBox[3] || interBox[0] >= interBox[2])
        return 0.f;

    float interBoxS = (interBox[2] - interBox[0]) * (interBox[3] - interBox[1]);

    float b1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    float b2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);

    return interBoxS / (b1_area + b2_area - interBoxS + 1e-16);
}

vector<vector<float>> convert_bbox_to_z(vector<int> bbox) {
    // [x1,y1,x2,y2] -> [x,y,s,r]^T
    vector<vector<float>> result(4, vector<float>(1));
    float w = bbox[2] - bbox[0];
    float h = bbox[3] - bbox[1];

    result[0][0] = (bbox[2] + bbox[0]) / 2.f;
    result[1][0] = (bbox[3] + bbox[1]) / 2.f;
    result[2][0] = w * h;
    result[3][0] = w / h;

    return result;
}

vector<float> convert_x_to_bbox(vector<vector<float>> x) {
    // [x,y,s,r] -> [x1,y1,x2,y2]
    vector<float> result;
    float w = sqrt(x[2][0] * x[3][0]);
    float h = x[2][0] / w;

    result.push_back(x[0][0] - w / 2.f);
    result.push_back(x[1][0] - h / 2.f);
    result.push_back(x[0][0] + w / 2.f);
    result.push_back(x[1][0] + h / 2.f);    

    return result;
}

void associate_detections_to_trackers(vector<vector<int>> dets, vector<vector<float>> trks, 
                                    vector<vector<int>>& matched, vector<int>& unmatched_dets, vector<int>& unmatched_trks) {
    if (trks.size() == 0) {
        for (size_t i = 0; i < dets.size(); ++i) {
            unmatched_dets.push_back(i);
        }
    }
    else {
        vector<vector<int>> matched_indices;
        vector<vector<float>> _iou_matrix(dets.size(), vector<float>(trks.size(), 0));
        for (size_t i = 0; i < _iou_matrix.size(); ++i) {
            for (size_t j = 0; j < _iou_matrix[0].size(); ++j) {
                _iou_matrix[i][j] = bbox_iou(dets[i], trks[j]) * -1;
            }
        }

        matched_indices = linear_sum_assignment(_iou_matrix);

        for (size_t i = 0; i < dets.size(); ++i) {
            if (matched_indices[0].size() == 0) {
                unmatched_dets.push_back(i);
            }
            else if (find(matched_indices[0].begin(), matched_indices[0].end(), i) == matched_indices[0].end()) {
                unmatched_dets.push_back(i);
            }
        }
        
        for (size_t i = 0; i < trks.size(); ++i) {
            if (matched_indices[1].size() == 0) {
                unmatched_trks.push_back(i);
            }
            else if (find(matched_indices[1].begin(), matched_indices[1].end(), i) == matched_indices[1].end()) {
                unmatched_trks.push_back(i);
            }
        }

        // filter out matched with low iou
        for (size_t i = 0; i < matched_indices[0].size(); ++i) {
            if (((-1 * _iou_matrix[matched_indices[0][i]][matched_indices[1][i]]) < IOU_THRESHOLD) || ((int)dets[matched_indices[0][i]][4] != (int)trks[matched_indices[1][i]][4])) {
                unmatched_dets.push_back(matched_indices[0][i]);
                unmatched_trks.push_back(matched_indices[1][i]);
            }
            else {
                matched[0].push_back(matched_indices[0][i]);
                matched[1].push_back(matched_indices[1][i]);
            }
        }
    }
}

KalmanBoxTracker::KalmanBoxTracker() {
}

void KalmanBoxTracker::initialize_variable(vector<int> bbox, int count){
    kf.initialize_variable(7, 4);

    kf.F = {{1, 0, 0, 0, 1, 0, 0}, {0, 1, 0, 0, 0, 1, 0}, {0, 0, 1, 0, 0, 0, 1}, {0, 0, 0, 1, 0, 0, 0},
            {0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 0, 1}};
    kf.H = {{1, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0}};
    kf.R[2][2] *= 10;
    kf.R[3][3] *= 10;
    for (int i = 4; i < 7; ++i) {
        kf.P[i][i] *= 1000;
    }
    for (int i = 0; i < 7; ++i) {
        kf.P[i][i] *= 10;
    }
    kf.Q[6][6] *= 0.01;
    for (int i = 4; i < 7; ++i) {
        kf.Q[i][i] *= 0.01;
    }

    vector<vector<float>> bbox2z = convert_bbox_to_z(bbox);
    kf.x[0][0] = bbox2z[0][0];
    kf.x[1][0] = bbox2z[1][0];
    kf.x[2][0] = bbox2z[2][0];
    kf.x[3][0] = bbox2z[3][0];
    id = count;
    time_since_update = 0;
    hits = 0;
    hit_streak = 0;
    age = 0;
}

void KalmanBoxTracker::update(vector<int> bbox) {
    time_since_update = 0;
    history.clear();
    hits += 1;
    hit_streak += 1;
    kf.update(convert_bbox_to_z(bbox));
}

vector<float> KalmanBoxTracker::predict() {
    if ((kf.x[6][0] + kf.x[2][0]) <= 0) kf.x[6][0] *= 0;
    kf.predict();
    age += 1;
    if (time_since_update > 0) hit_streak = 0;
    time_since_update += 1;
    history.push_back(convert_x_to_bbox(kf.x));

    return history[history.size() - 1];
}

vector<float> KalmanBoxTracker::get_state() {
    return convert_x_to_bbox(kf.x);
}

Sort::Sort() {
}

void Sort::initialize_variable(int age, int hits) {
    max_age = age;
    min_hits = hits;
    frame_count = 0;

    for (int i = 0; i < 3; ++i) {
		bbb.push_back(i);
	}
    vector<int> tmp;
    for (int i = 0; i < 5; ++i) {
        tmp.push_back(0);
    }
    aaa.now.push_back(tmp);
    aaa.minus1.push_back(tmp);
    aaa.minus2.push_back(tmp);
}

vector<vector<int>> Sort::update(vector<vector<int>> dets, int& count) {
    // dets: [x1,y1,x2,y2,type]
    // trks: [x1,y1,x2,y2,type]
    // ret:  [x1,y1,x2,y2,obj_id,type]

    for (int i = 1; i >= 0; --i) {
        bbb[i + 1] = bbb[i];
    }
    aaa.minus2 = aaa.minus1;
    aaa.minus1 = aaa.now;
    aaa.now = dets;
    bbb[0] = dets.size();

    frame_count += 1;
    vector<vector<float>> trks(trackers.size(), vector<float>());
    vector<int> to_del;
    vector<vector<int>> ret;

    vector<float> pos;
    vector<vector<int>> matched(2, vector<int> ());
    vector<int> unmatched_dets;
    vector<int> unmatched_trks;
    int del_flag = 0;

    for (size_t i = 0; i < trks.size(); ++i) {
        pos = trackers[i].predict();
        trks[i].push_back(pos[0]);
        trks[i].push_back(pos[1]);
        trks[i].push_back(pos[2]);
        trks[i].push_back(pos[3]);
        trks[i].push_back(types[i]);
        if (isnan(pos[0]) || isnan(pos[1]) || isnan(pos[2]) || isnan(pos[3])) {
            // 如果预测的bbox为空 删除第i个跟踪器
            to_del.push_back(i);
        }
    }
    // 删除预测为空的跟踪器所在行 (算的出nan吗。。。)
    for (vector<vector<float>>::iterator iter = trks.begin(); iter != trks.end(); ) {
        del_flag = 0;
        for (vector<float>::iterator it = (*iter).begin(); it != (*iter).end(); it++) {
            if (isnan(*it)) {
                del_flag = 1;
                break;
            }
        }
        if (del_flag) {
            iter = trks.erase(iter);
        }
        else {
            iter++;
        }
    }
    for (int i = to_del.size() - 1; i >= 0 ; i--) {
        trackers.erase(trackers.begin() + to_del[i]);
        types.erase(types.begin() + to_del[i]);
    }
    associate_detections_to_trackers(dets, trks, matched, unmatched_dets, unmatched_trks);

    // 遍历跟踪器 如果上一帧的i还在当前帧中 说明跟踪器i关联成功 
    // 在matched中找到与其关联的检测器d 用d更新卡尔曼跟踪器
    int d = 0;
    for (size_t i = 0; i < trackers.size(); ++i) {
        if (unmatched_trks.size() == 0) {
            for (size_t j = 0; j < matched[1].size(); ++j) {
                if (matched[1][j] == i) {
                    d = matched[0][j];
                    break;
                }
            }
            trackers[i].update(dets[d]);
            types[i] = dets[d][4];
        }
        else if (find(unmatched_trks.begin(), unmatched_trks.end(), i) == unmatched_trks.end())
        {
            for (size_t j = 0; j < matched[1].size(); ++j) {
                if (matched[1][j] == i) {
                    d = matched[0][j];
                    break;
                }
            }
            trackers[i].update(dets[d]);
            types[i] = dets[d][4];
        }
    }
    // 对于新增的未匹配的检测结果 创建初始化跟踪器track 并传入trackers
    int temp_trace = 0;
    vector<float> flag(unmatched_dets.size(), 0);
    for (size_t i = 0; i < unmatched_dets.size(); ++i) {
        for (int j = 0; j < bbb[1]; ++j) {
            if (bbox_iou(aaa.now[unmatched_dets[i]], aaa.minus1[j]) > IOU_THRESHOLD2 && aaa.now[unmatched_dets[i]][4] == aaa.minus1[j][4]) {
                flag[i] = 0.5;
                temp_trace = j;
                break;
            }
            flag[i] = -1;
        }
        if (flag[i] == 0.5) {
            for (int j = 0; j < bbb[2]; ++j) {
                if (bbox_iou(aaa.minus1[temp_trace], aaa.minus2[j]) > IOU_THRESHOLD2 && aaa.minus1[temp_trace][4] == aaa.minus2[j][4]) {
                    flag[i] = 1;
                    break;
                }
                flag[i] = -1;
            }
        }
        if (flag[i] == 1) {
            KalmanBoxTracker track;
            vector<int> tmp(dets[unmatched_dets[i]].begin(), dets[unmatched_dets[i]].begin() + 4);
            track.initialize_variable(tmp, count);
            count += 1;
            trackers.push_back(track);
            types.push_back(dets[unmatched_dets[i]][4]);
        }
    }

    for (int i = trackers.size() - 1; i >= 0 ; i--) {
        pos = trackers[i].get_state();
        if (trackers[i].time_since_update < 1 && (trackers[i].hit_streak >= min_hits || frame_count <= min_hits)) {
            vector<int> tmp_out;
            tmp_out.push_back((int)pos[0]);
            tmp_out.push_back((int)pos[1]);
            tmp_out.push_back((int)pos[2]);
            tmp_out.push_back((int)pos[3]);
            tmp_out.push_back(trackers[i].id + 1);
            tmp_out.push_back(types[i]);
            ret.push_back(tmp_out);
        }
        if (trackers[i].time_since_update > max_age) {
            trackers.erase(trackers.begin() + i);
            types.erase(types.begin() + i);
        }
    }

    return ret;
}