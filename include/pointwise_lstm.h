/*************************************************************************
    > File Name: lstm.h
    > Author: Wang Junqi
    > Mail: 351868656@qq.com 
    > Created Time: 2015年12月28日 星期一 13时18分36秒
 ************************************************************************/

#ifndef POINTWISE_LSTM_H
#define POINTWISE_LSTM_H

#include<iostream>
#include<vector>
#include"../lib/Eigen/Dense"

using namespace std;
using namespace Eigen;

class PointwiseLSTM
{

private:

    int input_cell_num;    //cell number of input layer(I)
    int hidden_cell_num;    //cell number of hidden layer(H)
    int state_cell_num;    //cell number of state cell(C)
    int output_num;    //cell number of output layer(K)
    int block_num;    //number of block (T)

    //weights matrixes
    MatrixXd state_input_gate_weights;    //the weights of input gate (for state cell) (C+1 * 1)
    MatrixXd hidden_input_gate_weights;    //the weights of input gate(for hidden cell) (H+1 * 1)
    MatrixXd input_input_gate_weights;    //the weights of input gate(for input cell) (I+1 * 1)
    MatrixXd state_output_gate_weights;    //the weights of output gate (for state cell) (C+1 * 1)
    MatrixXd hidden_output_gate_weights;    //the weights of output gate (for hidden cell) (H+1 * 1)
    MatrixXd input_output_gate_weights;    //the weights of output gate(for input cell) (I+1 * 1)

    MatrixXd state_forget_gate_weights;    //the weights of forget gate(for state cell) (C * C+1)
    MatrixXd hidden_forget_gate_weights;    //the weights of forget gate(for hidden cell) (C * H+1)
    MatrixXd input_forget_gate_weights;    //the weights of forget gate(for input cell) (C * I+1)

    MatrixXd input_input_tanh_weights;    //the weights of input tanh layer w.r.t. input layer (C * I+1)
    MatrixXd hidden_input_tanh_weights;    //the weights of input tanh layer w.r.t. hidden layer (C * H+1)
    MatrixXd output_tanh_weights;    //the weights of output tanh layer (H * C+1)
    MatrixXd output_weights;    //the weighs of output softmax layer (K * H+1)

    //memory matrixes
    MatrixXd state_values;    //all state value of time 1...T (C * T)
    MatrixXd hidden_values;    //all hidden value of tiem 1...T (H * T)
    MatrixXd input_values;    //the sequence of inputs (I * T)

    MatrixXd input_gate_act;    //the values of input gate activation for time 1...T (1 * T)
    MatrixXd output_gate_act;    //the values of output gate activation for time 1...T (1 * T)
    MatrixXd forget_gate_act;    //the values of forget gate activation for time 1...T (C * T)
    
    MatrixXd input_tanh_act;    //the values of input tanh layer activation for time 1...T (C * T)
    MatrixXd output_tanh_act;    //the values of output tanh layer activation for time 1...T (H * T)
    MatrixXd output_act;    //the values of output layer for time 1...T (K * T)

    //error matrixes
    MatrixXd error_output;    //the error of output layer (softmax layer) (K * T)
    MatrixXd error_hidden;    //the error of hidden layer (H * T)
    MatrixXd error_input_gate;    //the error of input gate (1 * T)
    MatrixXd error_output_gate;    //the error of output gate (1 * T)
    MatrixXd error_forget_gate;    //the error of forget gate (C * T)
    MatrixXd error_input_tanh;    //the errors of input tanh layer (C * T)
    MatrixXd error_state;    //the errors of state layer (C * T)
    MatrixXd error_output_tanh;    //the errors of output tanh layer (H * T)

public:

    PointwiseLSTM();    //default initialization
    PointwiseLSTM(string model_file);    //initialization with model stored in file
    PointwiseLSTM(int input_cell_num, int hidden_cell_num, int state_cell_num, int output_num);    //initialization with layers cell number
    void forwardPass(MatrixXd);    //forward psss geiven the input data (matrix)
    void calculateSoftMax(MatrixXd);    //calculate SoftMax for the output layer
    void backwardPass(MatrixXd&);    //calculate errors back through the network
    double calculateError(MatrixXd&);    //calculate the error of geiven input
    void checkGradient();    //check the derivative is calculated correct or not
    void updateWeightsWithOneSentence(
                                            MatrixXd &now_input_maxtrix, MatrixXd &now_label_maxtrix,
                                            double learning_rate, double momentum_rate,
                                            MatrixXd &state_input_gate_weights_derivative,
                                            MatrixXd &hidden_input_gate_weights_derivative,
                                            MatrixXd &input_input_gate_weights_derivative,
                                            MatrixXd &state_output_gate_weights_derivative,
                                            MatrixXd &hidden_output_gate_weights_derivative,
                                            MatrixXd &input_output_gate_weights_derivative,
                                            MatrixXd &state_forget_gate_weights_derivative,
                                            MatrixXd &hidden_forget_gate_weights_derivative,
                                            MatrixXd &input_forget_gate_weights_derivative,
                                            MatrixXd &input_input_tanh_weights_derivative,
                                            MatrixXd &hidden_input_tanh_weights_derivative,
                                            MatrixXd &output_tanh_weights_derivative,
                                            MatrixXd &output_weights_derivative
                                           );    //update network weights with one sentence
    void stochasticGradientDescent(vector<MatrixXd*>, vector<MatrixXd*>, double, double);    //SGD with a momentum
    vector<MatrixXd*> predict(vector<MatrixXd*>);    //predict the label of the geiven input datas
    void saveModel(string file_name);    //save current model into a file
    void loadModel(string file_name);    //load a model from geiven file

    //some getter function
    MatrixXd getOutputValue();    //get the matrix of output_act
    int getInputCellNum();    //get the value of input_cell_num
    int getStateCellNum();    //get the value of state_cell_num
    int getHiddenCellNum();    //get the value of hidden_cell_num
    int getOutputNum();    //get the value of output_num
};

#endif
