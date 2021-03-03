#include <iostream>

#include "hungarian.h"

Hungary::Hungary() {
}

void Hungary::initialize_variable(vector<vector<float>> cost_matrix) {
    Z0_r = 0;
    Z0_c = 0;
    C = cost_matrix;
    n = C.size();
    m = C[0].size();
    for (int i = 0; i < n; ++i) {
        row_uncovered.push_back(true);
    }
    for (int i = 0; i < m; ++i) {
        col_uncovered.push_back(true);
    }
    for (int i = 0; i < (n + m); ++i) {
        vector<int> tmp(2, 0);
        path.push_back(tmp);
    }
    for (int i = 0; i < n; ++i) {
        vector<int> tmp(m, 0);
        marked.push_back(tmp);
    }
    STEP_ID = 1;
}

void Hungary::clear_covers() {
    for (size_t i = 0; i < row_uncovered.size(); ++i) {
        row_uncovered[i] = true;
    }
    for (size_t i = 0; i < col_uncovered.size(); ++i) {
        col_uncovered[i] = true;
    }
}

void _step1(Hungary& state) {
    float min_r;
    for (int i = 0; i < state.n; ++i) {
        min_r = *min_element(state.C[i].begin(), state.C[i].end());
        for (int j = 0; j < state.m; ++j) {
            state.C[i][j] -= min_r;
        }
    }

    for (int i = 0; i < state.n; ++i) {
        for (int j = 0; j < state.m; ++j) {
            if (state.C[i][j] == 0 && state.col_uncovered[j] && state.row_uncovered[i]) {
                state.marked[i][j] = 1;
                state.col_uncovered[j] = false;
                state.row_uncovered[i] = false;
            }
        }
    }
    state.clear_covers();
    state.STEP_ID = 3;
}

void _step3(Hungary& state) {
    for (int i = 0; i < state.m; ++i) {
        for (int j = 0; j < state.n; ++j) {
            if (state.marked[j][i] == 1) {
                state.col_uncovered[i] = false;
                break;
            }
        }
    }
    int sum_marked = 0;
    for (int i = 0; i < state.n; ++ i) {
        for (int j = 0; j < state.m; ++ j) {
            sum_marked += state.marked[i][j];
        }
    }
    if (sum_marked < state.n) {
        state.STEP_ID = 4;
    }
    else {
        state.STEP_ID = 0;
    }
}

void _step4(Hungary& state) {
    vector<vector<int>> C_(state.n, vector<int> (state.m, 0));
    vector<vector<int>> cover_C(state.n, vector<int> (state.m, 0));
    for (int i = 0; i < state.n; ++i) {
        for (int j = 0; j < state.m; ++j) {
            if (state.C[i][j] == 0) C_[i][j] = 1;
            cover_C[i][j] = C_[i][j] * state.row_uncovered[i];
            cover_C[i][j] *= state.col_uncovered[j];
        }
    }
    while (true) {
        int row = 0;
        int col = 0;
        int cover_C_max = 0;
        for (int i = 0; i < state.n; ++i) {
            for (int j = 0; j < state.m; ++j) {
                if (cover_C[i][j] == 1 && cover_C_max == 0) {
                    row = i;
                    col = j;
                    cover_C_max = 1;
                }
            }
        }
        if (cover_C[row][col] == 0) {
            state.STEP_ID = 6;
            break;
        }
        else {
            state.marked[row][col] = 2;
            int star_col = 0;
            for (int i = 0; i < state.m; ++i) {
                if (state.marked[row][i] == 1) {
                    star_col = i;
                    break;
                }
            }
            if (state.marked[row][star_col] != 1) {
                state.Z0_r = row;
                state.Z0_c = col;
                state.STEP_ID = 5;
                break;
            }
            else {
                col = star_col;
                state.row_uncovered[row] = false;
                state.col_uncovered[col] = true;
                for (int i = 0; i < state.n; ++i) {
                    cover_C[i][col] = C_[i][col] * state.row_uncovered[i]; 
                }
                for (int i = 0; i < state.m; ++i) {
                    cover_C[row][i] = 0;
                }
            }
        }
    }
}

void _step5(Hungary& state) {
    int count = 0;
    vector<vector<int>> path_(state.path);
    path_[count][0] = state.Z0_r;
    path_[count][1] = state.Z0_c;
    while (true) {
        int row = 0;
        for (int i = 0; i < state.n; ++i) {
            if (state.marked[i][path_[count][1]] == 1) {
                row = i;
                break;
            }
        }
        if (state.marked[row][path_[count][1]] != 1) {
            break;
        }
        else {
            count += 1;
            path_[count][0] = row;
            path_[count][1] = path_[count - 1][1];
        }
        int col = 0;
        for (int i = 0; i < state.m; ++i) {
            if (state.marked[path_[count][0]][i] == 2) {
                col = i;
                break;
            }
        }
        if (state.marked[row][col] != 2) col = -1;
        count += 1;
        path_[count][0] = path_[count - 1][0];
        path_[count][1] = col;
    }
    for (int i = 0; i < (count + 1); ++i) {
        if (state.marked[path_[i][0]][path_[i][1]] == 1) {
            state.marked[path_[i][0]][path_[i][1]] = 0;
        }
        else {
            state.marked[path_[i][0]][path_[i][1]] = 1;
        }
    }
    state.clear_covers();
    for (int i = 0; i < state.n; ++i) {
        for (int j = 0; j < state.m; ++j) {
            if (state.marked[i][j] == 2) state.marked[i][j] = 0;
        }
    }
    state.STEP_ID = 3;
}

void _step6(Hungary& state) {
    if (find(state.row_uncovered.begin(), state.row_uncovered.end(), true) != state.row_uncovered.end()  && 
            find(state.col_uncovered.begin(), state.col_uncovered.end(), true) != state.col_uncovered.end()) {
        float minval = 1e+3;
        for (int i = 0; i < state.n; ++i) {
            if (state.row_uncovered[i]) {
                for (int j = 0; j < state.m; ++j) {
                    if (state.col_uncovered[j] && state.C[i][j] < minval) minval = state.C[i][j];
                }
            }
        }
        for (int i = 0; i < state.n; ++i) {
            if (!state.row_uncovered[i]) {
                for (int j = 0; j < state.m; ++j) {
                    state.C[i][j] += minval;
                }
            }
        }
        for (int i = 0; i < state.m; ++i) {
            if (state.col_uncovered[i]) {
                for (int j = 0; j < state.n; ++j) {
                    state.C[j][i] -= minval;
                }
            }
        }
    }
    state.STEP_ID = 4;
}

vector<vector<int>> linear_sum_assignment(vector<vector<float>> cost_matrix) {
    Hungary state;
    vector<vector<int>> results;
    vector<int> row_ind;
    vector<int> col_ind;

    if (cost_matrix.size() == 0) {
        state.marked = {};
    }
    else {
        bool transposed;
        if (cost_matrix[0].size() < cost_matrix.size()) {
            cost_matrix = matrix_transpose(cost_matrix);
            transposed = true;
        }
        else {
            transposed = false;
        }
        state.initialize_variable(cost_matrix);

        while (state.STEP_ID) {
            switch (state.STEP_ID) {
                case 1:
                    _step1(state);
                    break;
                case 3:
                    _step3(state);
                    break;
                case 4:
                    _step4(state);
                    break;
                case 5:
                    _step5(state);
                    break;
                case 6:
                    _step6(state);
                    break;
                default:
                    break;
            }
        }
        if (transposed) state.marked = matrix_transpose(state.marked);
    }

    for (size_t i = 0; i < state.marked.size(); ++i) {
        for (size_t j = 0; j < state.marked[0].size(); ++j) {
            if (state.marked[i][j] == 1) {
                row_ind.push_back(i);
                col_ind.push_back(j);
            }
        }
    }
    results.push_back(row_ind);
    results.push_back(col_ind);

    return results;
}