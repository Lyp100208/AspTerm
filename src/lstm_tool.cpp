/*************************************************************************
	> File Name: lstm_tool.cpp
	> Author: Wang Junqi
	> Mail: 351868656@qq.com 
	> Created Time: 2016年01月06日 星期三 14时36分59秒
 ************************************************************************/

#include<iostream>
#include<cmath>
#include"../include/lstm_tool.h"

using namespace std;

//sigmoid function
double LSTMTool::sigmoid(double x)
{
    double result = 1 / (1 + exp(-x));
    if (isnan(result))
    {
        cout << "sigmiod error -------- " << x << endl;
    }
    return result;
}

//tanh function
double LSTMTool::tanhX(double x)
{
    //Prevent overflow of results
    if (x > 50)
    {
        return 1;
    }
    else if (x < -50)
    {
        return -1;
    }
    
    double plus_exp = exp(x);
    double neg_exp = exp(-x);
    double result =  (plus_exp - neg_exp) / (plus_exp + neg_exp);
    if (isnan(result))
    {
        cout << "tanh error -------- " << x << endl;
    }
    return result;
}

//sigmoid function for matrix (coefficient-wise)
void LSTMTool::sigmoidforMatrix(MatrixXd &matrix)
{
    int row_num = matrix.rows();
    int col_num = matrix.cols();
    for (int i = 0; i < row_num; ++i)
    {
        for (int j = 0; j < col_num; ++j)
        {
            matrix(i, j) = sigmoid(matrix(i, j));
        }
    }
    return;
}

//tanh function for matrix (coefficient-wise)
void LSTMTool::tanhXforMatrix(MatrixXd &matrix)
{
    int row_num = matrix.rows();
    int col_num = matrix.cols();
    for (int i = 0; i < row_num; ++i)
    {
        for (int j = 0; j < col_num; ++j)
        {
            matrix(i, j) = tanhX(matrix(i, j));
        }
    }
    return;
}

//calculate softmax values for a vector
void LSTMTool::calculateSoftMax(MatrixXd &output)
{
    double sum = 0;
    for (int i = 0; i < output.rows(); ++i)
    {
        output(i, 0) = exp(output(i, 0));
        sum += output(i, 0);
    }
    for (int i = 0; i < output.rows(); ++i)
    {
        output(i, 0) = output(i, 0) / sum;
    }
    return;
}

//randomising the order of the geiven sequences
void LSTMTool::disturbOrder(vector<MatrixXd*> &vector_1, vector<MatrixXd*> &vector_2)
{
    if (vector_1.size() != vector_2.size())
    {
        return;
    }
    int sequence_length = vector_1.size();

    srand((unsigned int)time(NULL));
    for (int i = 0; i < sequence_length / 2; ++i)
    {
        int index_1 = rand() % sequence_length;
        int index_2 = rand() % sequence_length;
        MatrixXd* temp_pointer = NULL;

        //change the index of vector_1
        temp_pointer = vector_1[index_1];
        vector_1[index_1] = vector_1[index_2];
        vector_1[index_2] = temp_pointer;

        //change the index of vector_2
        temp_pointer = vector_2[index_1];
        vector_2[index_1] = vector_2[index_2];
        vector_2[index_2] = temp_pointer;
    }
    return;
}

//random Initialize geiven matrix
void LSTMTool::randomInitialize(MatrixXd &matrix, double precision)
{
    /*
     * argument : matrix -- matrix waiting to be initialized
     *            precision -- the range for value i.e (-precision, precision)
     */
    
    srand((unsigned)time(NULL));    //set time seed

    for (int row_num = 0; row_num < matrix.rows(); ++row_num)
    {
        for (int col_num = 0; col_num < matrix.cols(); ++col_num)
        {
            double random_val = 2 * rand() - RAND_MAX;
            matrix(row_num, col_num) = random_val / double(RAND_MAX) * precision;
        }
    }
    
    return;
}

//recerse the matrix in the row direction
void LSTMTool::reverseMarix(MatrixXd &temp_matrix)
{
    //reverse the matrix
    int forward_point = 0;
    int backward_point = temp_matrix.cols() - 1;
    while (backward_point > forward_point)
    {
        temp_matrix.col(forward_point) += temp_matrix.col(backward_point);
        temp_matrix.col(backward_point) = temp_matrix.col(forward_point) - temp_matrix.col(backward_point);
        temp_matrix.col(forward_point) = temp_matrix.col(forward_point) - temp_matrix.col(backward_point);
        ++forward_point;
        --backward_point;
    }

    return;
}

