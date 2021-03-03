#ifndef __KALMAN_FILTER_H_
#define __KALMAN_FILTER_H_

#include <iostream>
#include <vector>
#include <algorithm>

#include "matrix.h"

using namespace std;

class Kalman_Filter {

    public:
        int dim_x;
        int dim_z;

        vector<vector<float>> x;
        vector<vector<float>> P;
        vector<vector<float>> Q;
        vector<vector<float>> F;
        vector<vector<float>> H;
        vector<vector<float>> R;
        float _alpha_sq;
        vector<vector<float>> M;
        vector<vector<float>> z;

        vector<vector<float>> K;
        vector<vector<float>> y;
        vector<vector<float>> S;
        vector<vector<float>> SI;

        vector<vector<float>> _I;

        vector<vector<float>> x_prior;
        vector<vector<float>> P_prior;

        vector<vector<float>> x_post;
        vector<vector<float>> P_post;
        
    public:
        Kalman_Filter();
        ~Kalman_Filter() {
        }

        void initialize_variable(int dimx, int dimz);
        void update(vector<vector<float>> bbox);
        void predict();

};

#endif
