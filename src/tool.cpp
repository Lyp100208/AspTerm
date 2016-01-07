/*************************************************************************
	> File Name: tool.cpp
	> Author: Wang Junqi
	> Mail: 351868656@qq.com 
	> Created Time: 2016年01月06日 星期三 14时36分59秒
 ************************************************************************/

#include<iostream>
#include"../include/tool.h"

using namespace std;

//sigmoid function
double Tool::sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

//tanh function
double Tool::tanhX(double x)
{
    double plus_exp = exp(x);
    double neg_exp = exp(-x);
    return (plus_exp - neg_exp) / (plus_exp + neg_exp);
}

//tanh function for matrix (coefficient-wise)
void Tool::tanhXforMatrix(MatrixXd &matrix)
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
void Tool::calculateSoftMax(MatrixXd &output)
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
void Tool::disturbOrder(vector<MatrixXd*> &vector_1, vector<MatrixXd*> &vector_2)
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
