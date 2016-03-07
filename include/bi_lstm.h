/*************************************************************************
	> File Name: bi_lstm.h
	> Author: Wang Junqi
	> Mail: 351868656@qq.com 
	> Created Time: 2016年02月20日 星期六 18时12分21秒
 ************************************************************************/

#ifndef BI_LSTM_H
#define BI_LSTM_H

#include<iostream>
#include"../include/lstm.h"
using namespace std;

class BiLSTM
{
    
private:

    LSTM forward_lstm;
    LSTM backward_lstm;

public:
    
    BiLSTM();    //default initialization
    BiLSTM(string model_file);    //initialization with model stored in file
    BiLSTM(int input_cell_num, int hidden_cell_num, int state_cell_num, int output_num);    //initialization with layters cell number
    void forwardPass(MatrixXd);    //forward pass geiven the input data (I * T)
    void calculateSoftMax();    //calculate SoftMax layer with both forward and backward LSTM hidden layer
    void backwardPass(MatrixXd);    //calculate errors back through the network
    void stochasticGradientDescent(vector<MatrixXd*>, vector<MatrixXd*>, double, double);    //SGD with a momentum
    vector<MatrixXd*> predict(vector<MatrixXd*>);    //predict the label of the geiven input datas
    void saveModel(string);    //save current model into one file to store forward and backward LSMT
    void loadModel(string);    //load a model (forward and backward LSTM) from geiven file

};

#endif
