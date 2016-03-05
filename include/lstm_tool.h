/*************************************************************************
	> File Name: tool.h
	> Author: Wang Junqi
	> Mail: 351868656@qq.com 
	> Created Time: 2016年01月06日 星期三 14时21分26秒
 ************************************************************************/

#ifndef LSTM_TOOL_H
#define LSTM_TOOL_H

#include<iostream>
#include<vector>
#include"../lib/Eigen/Dense"

using namespace Eigen;
using namespace std;

class LSTMTool
{

public:
    
    static double sigmoid(double);    //sigmoid function
    static double tanhX(double);    //tanh function 
    static void tanhXforMatrix(MatrixXd&);    //tanh function for matrix (coefficient-wise)
    static void calculateSoftMax(MatrixXd&);    //calculate softmax values for a vector
    static void disturbOrder(vector<MatrixXd*>&, vector<MatrixXd*>&);    //randomising the order of the geiven sequences
    static void randomInitialize(MatrixXd&, double);    //random Initialize geiven matrix
    static void reverseMarix(MatrixXd&);    //reverse the matrix in the row direction
    static void pointwiseMult(MatrixXd&, MatrixXd&);    //pointwise multiplication for two matrix

};

#endif
