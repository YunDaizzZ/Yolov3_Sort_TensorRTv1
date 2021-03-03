#include <iostream>
#include <vector>
#include <algorithm>

#include "kalmanfilter.h"

using namespace std;

Kalman_Filter::Kalman_Filter() {
}

void Kalman_Filter::initialize_variable(int dimx, int dimz) {
    dim_x = dimx;
    dim_z = dimz;
    _alpha_sq = 1.f;

    for (int i = 0; i < dim_x; ++i) {
        vector<float> tmp1(dim_x, 0);
        tmp1[i] = 1.f;
        P.push_back(tmp1);
        Q.push_back(tmp1);
        F.push_back(tmp1);
        _I.push_back(tmp1);
        vector<float> tmp2(dim_z, 0);
        K.push_back(tmp2);
        vector<float> tmp3(1, 0);
        x.push_back(tmp3);
    }
    for (int i = 0; i < dim_z; ++i) {    
        vector<float> tmp1(dim_x, 0);
        H.push_back(tmp1);
        vector<float> tmp2(dim_z, 0);
        M.push_back(tmp2);
        S.push_back(tmp2);
        SI.push_back(tmp2);
        tmp2[i] = 1.f;
        R.push_back(tmp2);
        vector<float> tmp3(1, 0);
        y.push_back(tmp3);
    }

    x_prior = x;
    P_prior = P;
    x_post = x;
    P_post = P;
}

void Kalman_Filter::predict() {
    // x' = Fx + Bu  暂默认没有Bu
    x = dot(F, x);
    // P' = FPF^T + Q
    P = matrix_add(matrix_times(dot(dot(F, P), matrix_transpose(F)), _alpha_sq), Q);

    x_prior = x;
    P_prior = P;
}

void Kalman_Filter::update(vector<vector<float>> bbox) {
    if (bbox.size() == 0) {
        z.clear();
        x_post = x;
        P_post = P;
        y.clear();
        for (int i = 0; i < dim_z; ++i) {    
            vector<float> tmp3(1, 0);
            y.push_back(tmp3);
        }
    }
    else {
        // y = z - Hx'
        y = matrix_minus(bbox, dot(H, x));
        vector<vector<float>> PHT = dot(P, matrix_transpose(H));
        // S = HP'H^T + R
        S = matrix_add(dot(H, PHT), R);
        SI = inverse(S);
        // K = P'H^TS^-1
        K = dot(PHT, SI);
        // x = x' + Ky
        x = matrix_add(x, dot(K, y));
        // P = (I - KH)P' 或 P = (I - KH)P'(I - KH)^T + KRK^T
        vector<vector<float>> I_KH = matrix_minus(_I, dot(K, H));
        P = matrix_add(dot(dot(I_KH, P), matrix_transpose(I_KH)), dot(dot(K, R), matrix_transpose(K)));
        z = bbox;
        x_post = x;
        P_post = P;
    }
}


