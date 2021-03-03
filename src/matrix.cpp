#include <iostream>

#include "matrix.h"

vector<vector<float>> matrix_transpose(vector<vector<float>> A) {
    int rows = A.size();
    int cols = A[0].size();
    vector<vector<float>> v(cols, vector<float>());
    if (A.empty()) return vector<vector<float>>();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            v[j].push_back(A[i][j]);
        }
    }
    return v;
}

vector<vector<int>> matrix_transpose(vector<vector<int>> A) {
    int rows = A.size();
    int cols = A[0].size();
    vector<vector<int>> v(cols, vector<int>());
    if (A.empty()) return vector<vector<int>>();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            v[j].push_back(A[i][j]);
        }
    }
    return v;
}

vector<vector<float>> dot(vector<vector<float>> A, vector<vector<float>> B) {
    int rows_A = A.size();
    int cols_A = A[0].size();
    int cols_B = B[0].size();
    vector<vector<float>> v(rows_A, vector<float>(cols_B));
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_A; ++j) {
            for (int k = 0; k < cols_B; ++k) {
                v[i][k] += A[i][j] * B[j][k];
            }
        }
    }
    return v;
}

vector<vector<float>> matrix_add(vector<vector<float>> A, vector<vector<float>> B) {
    int rows_A = A.size();
    int cols_A = A[0].size();
    vector<vector<float>> v(A);
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_A; ++j) {
            v[i][j] += B[i][j];
        }
    }
    return v;
}

vector<vector<float>> matrix_minus(vector<vector<float>> A, vector<vector<float>> B) {
    int rows_A = A.size();
    int cols_A = A[0].size();
    vector<vector<float>> v(A);
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_A; ++j) {
            v[i][j] -= B[i][j];
        }
    }
    return v;
}

vector<vector<float>> matrix_times(vector<vector<float>> A, float B) {
    int rows_A = A.size();
    int cols_A = A[0].size();
    vector<vector<float>> v(A);
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_A; ++j) {
            v[i][j] *= B;
        }
    }
    return v;
}

vector<vector<float>> inverse(vector<vector<float>> A) {
    int N = A.size();
    float s, t;
    vector<vector<float>> v(N, vector<float>(N));
    vector<vector<float>> L(N, vector<float>(N));
    vector<vector<float>> U(N, vector<float>(N));

    vector<vector<float>> r(N, vector<float>(N));
    vector<vector<float>> u(N, vector<float>(N));

    // https://blog.csdn.net/xx_123_1_rj/article/details/39553809 那张图
    // 求L、U
    for (int j = 0; j < N; ++j) {
        A[0][j] = A[0][j];
    }
    for (int i = 1; i < N; ++i) {
        A[i][0] /= A[0][0];
    }
    for (int k = 1; k < N; ++k) {
        for (int j = k; j < N; ++j) {
            s = 0;
            for (int i = 0; i < k; ++i) {
                s += A[k][i] * A[i][j];
            }
            A[k][j] -= s;
        }
        for (int i = k + 1; i < N; ++i) {
            t = 0;
            for (int j = 0; j < k; ++j) {
                t += A[i][j] * A[j][k];
            }
            A[i][k] = (A[i][k] - t) / A[k][k];
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i > j) {
                L[i][j] = A[i][j];
                U[i][j] = 0;
            }
            else {
                U[i][j] = A[i][j];
                if (i == j) {
                    L[i][j] = 1;
                }
                else {
                    L[i][j] = 0;
                }
            }
        }
    }
    // 求L、U的逆r、u
    for (int i = 0; i < N; ++i) {
        u[i][i] = 1.f / U[i][i];
        for (int k = i - 1; k >= 0; k--) {
            s = 0;
            for (int j = k + 1; j <= i; ++j) {
                s += U[k][j] * u[j][i];
            }
            u[k][i] = -s / U[k][k];
        }
    }
    for (int i = 0; i < N; ++i) {
        r[i][i] = 1;
        for (int k = i + 1; k < N; ++k) {
            for (int j = i; j <= k - 1; ++j) {
                r[k][i] -= L[k][j] * r[j][i];
            }
        }
    }
    v = dot(u, r);
    return v;
}