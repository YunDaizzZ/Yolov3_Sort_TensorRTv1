#ifndef __HUNGARIAN_H_
#define __HUNGARIAN_H_

#include <iostream>
#include <vector>
#include <algorithm>

#include "matrix.h"

using namespace std;

class Hungary {

    public:
        vector<vector<float>> C;
        int n;
        int m;
        vector<bool> row_uncovered;
        vector<bool> col_uncovered;
        int Z0_r;
        int Z0_c;
        vector<vector<int>> path;
        vector<vector<int>> marked;
        int STEP_ID;

    public:
        Hungary();
        ~Hungary() {
        }

        void initialize_variable(vector<vector<float>> cost_matrix);
        void clear_covers();
};

void _step1(Hungary& state);
void _step3(Hungary& state);
void _step4(Hungary& state);
void _step5(Hungary& state);
void _step6(Hungary& state);

vector<vector<int>> linear_sum_assignment(vector<vector<float>> cost_matrix);

#endif