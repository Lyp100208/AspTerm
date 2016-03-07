/*************************************************************************
	> File Name: bi_lstm.cpp
	> Author: Wang Junqi
	> Mail: 351868656@qq.com 
	> Created Time: 2016年02月20日 星期六 18时33分48秒
 ************************************************************************/

#include<iostream>
#include"../include/bi_lstm.h"
#include"../include/lstm_tool.h"

using namespace std;

//default initialization
BiLSTM::BiLSTM()
{
    return;
}

//initialization with model stored in file
BiLSTM::BiLSTM(string model_file)
{
    loadModel(model_file);
    return;
}

//initialization with layers cell number
BiLSTM::BiLSTM(int input_cell_num, int hidden_cell_num, int state_cell_num, int output_num) : 
    forward_lstm(input_cell_num, hidden_cell_num, state_cell_num, output_num),
    backward_lstm(input_cell_num, hidden_cell_num, state_cell_num, output_num)
{
    return;
}

//forward pass geiven  the input data (I * T)
void BiLSTM::forwardPass(MatrixXd input_data)
{
    /*
     * argument : input_data -- the input sequence stored in matrix (I * T)
     */
    //the forward LSTM forward pass
    forward_lstm.forwardPass(input_data);

    //reverse the data matrix
    LSTMTool::reverseMarix(input_data);
    
    //the backward LSTM forward pass
    backward_lstm.forwardPass(input_data);
    return;
}

//calculate SoftMax layer with both forward and backward LSTM hidden layer
void BiLSTM::calculateSoftMax()
{
    /*
     * note : this function add forward and backward output layer together, and then calculate the softmax
     */

    MatrixXd temp_forward_matrix = forward_lstm.getOutputValue();
    MatrixXd temp_backward_matrix = backward_lstm.getOutputValue();
    
    //reverse the backward matrix
    LSTMTool::reverseMarix(temp_backward_matrix);

    MatrixXd temp_matrix = temp_forward_matrix + temp_backward_matrix;
    forward_lstm.calculateSoftMax(temp_matrix);
    
    LSTMTool::reverseMarix(temp_matrix);
    backward_lstm.calculateSoftMax(temp_matrix);
    return;
}

//calculate errors back through the network
void BiLSTM::backwardPass(MatrixXd label)
{
    /*
     * note : this function should be invoked after forwardPass() and calculateSoftMax()
     * argument : label -- the labels of input datas (K * T)
     * function : calculate errors back through the network
     */
    
    //the forward LSTM backward pass
    forward_lstm.backwardPass(label);
    
    //reverse the label matrix
    LSTMTool::reverseMarix(label);

    //the backward LSTM backward pass
    backward_lstm.backwardPass(label);

    return;
}

//SGD with a momentum
void BiLSTM::stochasticGradientDescent(vector<MatrixXd*> input_datas, vector<MatrixXd*> input_labels, double learning_rate, double momentum_rate)
{
    /*
     * note : the stopping criteria is 
     * argument : input_datas -- a vector of input data, every element is a I * T matrix
     *            input_labels -- a vector of input labels, every element is a K * T matrix
     *            learning_rate -- initial learning rate
     *            momentum_rate -- the parameter of momentum
     */

    //matrixes record previous derivative for forward LSTM
    MatrixXd forward_state_input_gate_weights_derivative = MatrixXd::Zero(forward_lstm.getStateCellNum() + 1, 1);
    MatrixXd forward_hidden_input_gate_weights_derivative = MatrixXd::Zero(forward_lstm.getHiddenCellNum() + 1, 1);
    MatrixXd forward_input_input_gate_weights_derivative = MatrixXd::Zero(forward_lstm.getInputCellNum() + 1, 1);
    MatrixXd forward_state_output_gate_weights_derivative = MatrixXd::Zero(forward_lstm.getStateCellNum() + 1, 1);
    MatrixXd forward_hidden_output_gate_weights_derivative = MatrixXd::Zero(forward_lstm.getHiddenCellNum() + 1, 1);
    MatrixXd forward_input_output_gate_weights_derivative = MatrixXd::Zero(forward_lstm.getInputCellNum() + 1, 1);
    MatrixXd forward_state_forget_gate_weights_derivative = MatrixXd::Zero(forward_lstm.getStateCellNum() + 1, 1);
    MatrixXd forward_hidden_forget_gate_weights_derivative = MatrixXd::Zero(forward_lstm.getHiddenCellNum() + 1, 1);
    MatrixXd forward_input_forget_gate_weights_derivative = MatrixXd::Zero(forward_lstm.getInputCellNum() + 1, 1);

    MatrixXd forward_input_input_tanh_weights_derivative = MatrixXd::Zero(forward_lstm.getStateCellNum(), forward_lstm.getInputCellNum() + 1);
    MatrixXd forward_hidden_input_tanh_weights_derivative = MatrixXd::Zero(forward_lstm.getStateCellNum(), forward_lstm.getHiddenCellNum() + 1);
    MatrixXd forward_output_tanh_weights_derivative = MatrixXd::Zero(forward_lstm.getHiddenCellNum(), forward_lstm.getStateCellNum() + 1);
    MatrixXd forward_output_weights_derivative = MatrixXd::Zero(forward_lstm.getOutputNum(), forward_lstm.getHiddenCellNum() + 1);

    //matrixes record previous derivative for backward LSTM
    MatrixXd backward_state_input_gate_weights_derivative = MatrixXd::Zero(backward_lstm.getStateCellNum() + 1, 1);
    MatrixXd backward_hidden_input_gate_weights_derivative = MatrixXd::Zero(backward_lstm.getHiddenCellNum() + 1, 1);
    MatrixXd backward_input_input_gate_weights_derivative = MatrixXd::Zero(backward_lstm.getInputCellNum() + 1, 1);
    MatrixXd backward_state_output_gate_weights_derivative = MatrixXd::Zero(backward_lstm.getStateCellNum() + 1, 1);
    MatrixXd backward_hidden_output_gate_weights_derivative = MatrixXd::Zero(backward_lstm.getHiddenCellNum() + 1, 1);
    MatrixXd backward_input_output_gate_weights_derivative = MatrixXd::Zero(backward_lstm.getInputCellNum() + 1, 1);
    MatrixXd backward_state_forget_gate_weights_derivative = MatrixXd::Zero(backward_lstm.getStateCellNum() + 1, 1);
    MatrixXd backward_hidden_forget_gate_weights_derivative = MatrixXd::Zero(backward_lstm.getHiddenCellNum() + 1, 1);
    MatrixXd backward_input_forget_gate_weights_derivative = MatrixXd::Zero(backward_lstm.getInputCellNum() + 1, 1);

    MatrixXd backward_input_input_tanh_weights_derivative = MatrixXd::Zero(backward_lstm.getStateCellNum(), backward_lstm.getInputCellNum() + 1);
    MatrixXd backward_hidden_input_tanh_weights_derivative = MatrixXd::Zero(backward_lstm.getStateCellNum(), backward_lstm.getHiddenCellNum() + 1);
    MatrixXd backward_output_tanh_weights_derivative = MatrixXd::Zero(backward_lstm.getHiddenCellNum(), backward_lstm.getStateCellNum() + 1);
    MatrixXd backward_output_weights_derivative = MatrixXd::Zero(backward_lstm.getOutputNum(), backward_lstm.getHiddenCellNum() + 1);

    for (int iteration_num = 0; iteration_num < 1; ++iteration_num)
    {

    if (iteration_num == 5)
    {
        learning_rate *= 0.1;
    }

    //randomising the order of input datas and labels
    LSTMTool::disturbOrder(input_datas, input_labels);

    for (int input_index = 0; input_index < input_datas.size(); ++input_index)
    {
        MatrixXd now_input_maxtrix = *(input_datas[input_index]);    //current input data
        MatrixXd now_label_maxtrix = *(input_labels[input_index]);    //current input label

        forwardPass(now_input_maxtrix);
        calculateSoftMax();
        backwardPass(now_label_maxtrix);

        //update weights for forward LSTM
        forward_lstm.updateWeightsWithOneSentence
        (
            now_input_maxtrix, now_label_maxtrix,
            learning_rate, momentum_rate,
            forward_state_input_gate_weights_derivative,
            forward_hidden_input_gate_weights_derivative,
            forward_input_input_gate_weights_derivative,
            forward_state_output_gate_weights_derivative,
            forward_hidden_output_gate_weights_derivative,
            forward_input_output_gate_weights_derivative,
            forward_state_forget_gate_weights_derivative,
            forward_hidden_forget_gate_weights_derivative,
            forward_input_forget_gate_weights_derivative,
            forward_input_input_tanh_weights_derivative,
            forward_hidden_input_tanh_weights_derivative,
            forward_output_tanh_weights_derivative,
            forward_output_weights_derivative
        );

        //reverse data matrix and label matrix
        LSTMTool::reverseMarix(now_input_maxtrix);
        LSTMTool::reverseMarix(now_label_maxtrix);

        //update weights for backward LSTM
        backward_lstm.updateWeightsWithOneSentence
        (
            now_input_maxtrix, now_label_maxtrix,
            learning_rate, momentum_rate,
            backward_state_input_gate_weights_derivative,
            backward_hidden_input_gate_weights_derivative,
            backward_input_input_gate_weights_derivative,
            backward_state_output_gate_weights_derivative,
            backward_hidden_output_gate_weights_derivative,
            backward_input_output_gate_weights_derivative,
            backward_state_forget_gate_weights_derivative,
            backward_hidden_forget_gate_weights_derivative,
            backward_input_forget_gate_weights_derivative,
            backward_input_input_tanh_weights_derivative,
            backward_hidden_input_tanh_weights_derivative,
            backward_output_tanh_weights_derivative,
            backward_output_weights_derivative
        );

    }    // end 'for' of input_datas

    }    //end 'for' of iteration_num
    return;
}

//predict the label of the geiven input datas
vector<MatrixXd*> BiLSTM::predict(vector<MatrixXd*> input_datas)
{
    /*
     * argument : input_datas -- the data waiting to be predict, every element is a matrix pointer (I * T)
     * return : predict_labels -- the label of input datas predicted by LSTM, every element is a matrix pointer (K * T)
     */
    
    vector<MatrixXd*> predict_labels;    //vector to store the predict labels

    //for every input predict the label
    for (int input_index = 0; input_index < input_datas.size(); ++input_index)
    {
        MatrixXd temp_input_data = *(input_datas[input_index]);
         
        forwardPass(temp_input_data);
        calculateSoftMax();
        MatrixXd temp_output_act = forward_lstm.getOutputValue();

        int output_num = forward_lstm.getOutputNum();
        int sequence_length = temp_input_data.cols();

        MatrixXd *temp_predict_label = new MatrixXd(output_num, sequence_length);
        *temp_predict_label = MatrixXd::Zero(output_num, sequence_length);

        
        for (int col_num = 0; col_num < sequence_length; ++col_num)
        {
            int temp_row = 0;
            int temp_col = 0;
            int *p_temp_row = &temp_row;
            int *p_temp_col = &temp_col;
            temp_output_act.col(col_num).maxCoeff(p_temp_row, p_temp_col);
            (*temp_predict_label)(*p_temp_row, col_num) = 1;
        }

        predict_labels.push_back(temp_predict_label);
    }
    return predict_labels;
}

//save the model into a file
void BiLSTM::saveModel(string file_name)
{
    /*
     * note : the forward LSTM and the backward LSTM will be writen into two file with diffirent postfix (file_name + postfix)
     * argument : file_name -- the prefix to store the model
     */
    
    forward_lstm.saveModel(file_name + "_forward_LSTM");
    backward_lstm.saveModel(file_name + "_backward_LSTM");

    return;
}

//load model from geiven file
void BiLSTM::loadModel(string file_name)
{
    /*
     * note : the forward LSTM and backward LSTM will be readed from two file with diffirent postfix (file_name + postfix)
     * argument : file_name -- the prefix of the file to store the model
     */

    forward_lstm.loadModel(file_name + "_forward_LSTM");
    backward_lstm.loadModel(file_name + "_backward_LSTM");

    return;
}
