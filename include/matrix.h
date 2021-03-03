#ifndef __MATRIX_H_
#define __MATRIX_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace std;

vector<vector<float>> matrix_transpose(vector<vector<float>> A);
vector<vector<int>> matrix_transpose(vector<vector<int>> A);

vector<vector<float>> dot(vector<vector<float>> A, vector<vector<float>> B);

vector<vector<float>> matrix_add(vector<vector<float>> A, vector<vector<float>> B);

vector<vector<float>> matrix_minus(vector<vector<float>> A, vector<vector<float>> B);

vector<vector<float>> matrix_times(vector<vector<float>> A, float B);

vector<vector<float>> inverse(vector<vector<float>> A);

#endif