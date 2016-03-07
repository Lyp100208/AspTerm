/*************************************************************************
    > File Name: pointwise_lstm.cpp
    > Author: Wang Junqi
    > Mail: 351868656@qq.com 
    > Created Time: 2016年2月29日 星期二 15时04分16秒
 ************************************************************************/

#include<iostream>
#include<cmath>
#include<fstream>

#include"../include/pointwise_lstm.h"
#include"../include/lstm_tool.h"

using namespace std;

//default initialization
PointwiseLSTM::PointwiseLSTM()
{
    return;
}

//initialization with model stored in file
PointwiseLSTM::PointwiseLSTM(string model_file)
{
    loadModel(model_file);
    return;
}

//initialization with layers cell number
PointwiseLSTM::PointwiseLSTM(int input_cell_num, int hidden_cell_num, int state_cell_num, int output_num) : 

    state_input_gate_weights(state_cell_num + 1, 1),
    hidden_input_gate_weights(hidden_cell_num + 1, 1),
    input_input_gate_weights(input_cell_num + 1, 1),
    state_output_gate_weights(state_cell_num + 1, 1),
    hidden_output_gate_weights(hidden_cell_num + 1, 1),
    input_output_gate_weights(input_cell_num + 1, 1),
    state_forget_gate_weights(state_cell_num, state_cell_num + 1),
    hidden_forget_gate_weights(state_cell_num, hidden_cell_num + 1),
    input_forget_gate_weights(state_cell_num, input_cell_num + 1),
    input_input_tanh_weights(state_cell_num, input_cell_num + 1),
    hidden_input_tanh_weights(state_cell_num, hidden_cell_num + 1),
    output_tanh_weights(hidden_cell_num, state_cell_num + 1),
    output_weights(output_num, hidden_cell_num + 1)

{
    this -> input_cell_num = input_cell_num;
    this -> hidden_cell_num = hidden_cell_num;
    this -> state_cell_num = state_cell_num;
    this -> output_num = output_num;
    
    /*
    //initialize weights randomly from [-0.1, 0.1], Pseudo Random
    state_input_gate_weights = MatrixXd::Random(state_cell_num + 1, 1) * 0.1;
    hidden_input_gate_weights = MatrixXd::Random(hidden_cell_num + 1, 1) * 0.1;
    input_input_gate_weights = MatrixXd::Random(input_cell_num + 1, 1) * 0.1;
    state_output_gate_weights = MatrixXd::Random(state_cell_num + 1, 1) * 0.1;
    hidden_output_gate_weights = MatrixXd::Random(hidden_cell_num + 1, 1) * 0.1;
    input_output_gate_weights = MatrixXd::Random(input_cell_num + 1, 1) * 0.1;
    state_forget_gate_weights = MatrixXd::Random(state_cell_num + 1, 1) * 0.1;
    hidden_forget_gate_weights = MatrixXd::Random(hidden_cell_num + 1, 1) * 0.1;
    input_forget_gate_weights = MatrixXd::Random(input_cell_num + 1, 1) * 0.1;
    input_input_tanh_weights = MatrixXd::Random(state_cell_num, input_cell_num + 1) * 0.1;
    hidden_input_tanh_weights = MatrixXd::Random(state_cell_num, hidden_cell_num + 1) * 0.1;
    output_tanh_weights = MatrixXd::Random(hidden_cell_num, state_cell_num + 1) * 0.1;
    output_weights = MatrixXd::Random(output_num, hidden_cell_num + 1) * 0.1;
    */

    //initialize weigths randomly with gieven precision, True Random
    LSTMTool::randomInitialize(state_input_gate_weights, 0.1);
    LSTMTool::randomInitialize(hidden_input_gate_weights, 0.1);
    LSTMTool::randomInitialize(input_input_gate_weights, 0.1);
    LSTMTool::randomInitialize(state_output_gate_weights, 0.1);
    LSTMTool::randomInitialize(hidden_output_gate_weights, 0.1);
    LSTMTool::randomInitialize(input_output_gate_weights, 0.1);
    LSTMTool::randomInitialize(state_forget_gate_weights, 0.1);
    LSTMTool::randomInitialize(hidden_forget_gate_weights, 0.1);
    LSTMTool::randomInitialize(input_forget_gate_weights, 0.1);
    LSTMTool::randomInitialize(input_input_tanh_weights, 0.1);
    LSTMTool::randomInitialize(hidden_input_tanh_weights, 0.1);
    LSTMTool::randomInitialize(output_tanh_weights, 0.1);
    LSTMTool::randomInitialize(output_weights, 0.1);

    return;
}

//forward pass geiven the input data (matrix)
void PointwiseLSTM::forwardPass(MatrixXd temp_input_matrix)
{
    /*
     * input_matrix : input sequence (not added bias) (I * T)
     */
    
    int sequence_length = temp_input_matrix.cols();    //sequence length
    block_num = sequence_length;    //record in variable 'block_num'

    //initialize memory matrix with zero
    state_values = MatrixXd::Zero(state_cell_num, sequence_length);
    hidden_values = MatrixXd::Zero(hidden_cell_num, sequence_length);
    input_gate_act = MatrixXd::Zero(1, sequence_length);
    output_gate_act = MatrixXd::Zero(1, sequence_length);
    forget_gate_act = MatrixXd::Zero(state_cell_num, sequence_length);
    input_tanh_act = MatrixXd::Zero(state_cell_num, sequence_length);
    output_tanh_act = MatrixXd::Zero(hidden_cell_num, sequence_length);
    output_act = MatrixXd::Zero(output_num, sequence_length);

    //add bias for input data
    int input_row_num = temp_input_matrix.rows();
    int input_col_num = temp_input_matrix.cols();
    MatrixXd input_matrix(input_row_num + 1, input_col_num);
    input_matrix.block(0, 0, input_row_num, input_col_num) = temp_input_matrix;
    input_matrix.block(input_row_num, 0, 1, input_col_num) = MatrixXd::Ones(1, input_col_num);

    //for every input, calculate correlative values of inner cells
    for(int col_index = 0; col_index < sequence_length; ++col_index)
    {
    	const MatrixXd &temp_input = input_matrix.col(col_index);
    	const MatrixXd temp_input_trans = temp_input.transpose();
    	MatrixXd temp_hidden_trans(1, hidden_cell_num + 1);
    	MatrixXd temp_state_trans(1, state_cell_num + 1);
    	if (col_index == 0)
    	{
    		temp_hidden_trans.block(0, 0, 1, hidden_cell_num) = MatrixXd::Zero(1, hidden_cell_num);    //initialization with all zero
    		temp_state_trans.block(0, 0, 1, state_cell_num) = MatrixXd::Zero(1, state_cell_num);    //initialization with all zero
    	}
    	else
    	{
    		temp_hidden_trans.block(0, 0, 1, hidden_cell_num) = hidden_values.col(col_index - 1).transpose();    //previous hidden cells value
    		temp_state_trans.block(0, 0, 1, state_cell_num) = state_values.col(col_index - 1).transpose();    //previous state cell values
    	}
    	temp_hidden_trans(0, hidden_cell_num) = 1;    //add bias term
    	temp_state_trans(0, state_cell_num) = 1;    //add bias term
    	
    	double temp_state_result;    //store the values of state vector * correlative weights vector
    	double temp_hidden_result;    //store the values of hidden vector * correlative weights vector
    	double temp_input_result;    //store the values of input vector * correlative weights vector

    	//calculate the activation of forget gate
        MatrixXd temp_state_forget_gate_val = state_forget_gate_weights * temp_state_trans.transpose();
        MatrixXd temp_hidden_forget_gate_val = hidden_forget_gate_weights * temp_hidden_trans.transpose();
        MatrixXd temp_input_forget_gete_val = input_forget_gate_weights * temp_input;
        MatrixXd temp_forget_gate_val = temp_state_forget_gate_val + temp_hidden_forget_gate_val + temp_input_forget_gete_val;
        LSTMTool::sigmoidforMatrix(temp_forget_gate_val);
        forget_gate_act.col(col_index) = temp_forget_gate_val;
    	
    	//calculate the activation of input gate
    	temp_state_result = (temp_state_trans * state_input_gate_weights)(0, 0);
    	temp_hidden_result = (temp_hidden_trans * hidden_input_gate_weights)(0, 0);
    	temp_input_result = (temp_input_trans * input_input_gate_weights)(0, 0);
    	double temp_input_gate_val = LSTMTool::sigmoid(temp_state_result + temp_hidden_result + temp_input_result);
    	input_gate_act(0, col_index) = temp_input_gate_val;

    	//calculate input tanh layer activation
    	MatrixXd input_input_tanh_result = input_input_tanh_weights * temp_input;
        MatrixXd hidden_input_tanh_result = hidden_input_tanh_weights * temp_hidden_trans.transpose();
        MatrixXd temp_input_tanh_result = input_input_tanh_result + hidden_input_tanh_result;
        LSTMTool::tanhXforMatrix(temp_input_tanh_result);
        input_tanh_act.col(col_index) = temp_input_tanh_result;
        
        //calculate new state cells values
        MatrixXd new_state_result = temp_forget_gate_val.array() * temp_state_trans.transpose().block(0, 0, state_cell_num, 1).array();
        new_state_result += temp_input_gate_val * temp_input_tanh_result;
        state_values.col(col_index) = new_state_result;
        MatrixXd new_state_trans(1, state_cell_num + 1);
        new_state_trans.block(0, 0, 1, state_cell_num) = new_state_result.transpose();
        new_state_trans(0, state_cell_num) = 1;    //add bias

    	//calculate the activation of output gate
        temp_state_result = (new_state_trans * state_output_gate_weights)(0, 0);
        temp_hidden_result = (temp_hidden_trans * hidden_output_gate_weights)(0, 0);
        temp_input_result = (temp_input_trans * input_output_gate_weights)(0, 0);
        double temp_output_gate_val = LSTMTool::sigmoid(temp_state_result + temp_hidden_result + temp_input_result);
        output_gate_act(0, col_index) = temp_output_gate_val;

        //calculate output tanh layer
        MatrixXd temp_output_tanh_result = output_tanh_weights * new_state_trans.transpose();
        LSTMTool::tanhXforMatrix(temp_output_tanh_result);
        output_tanh_act.col(col_index) = temp_output_tanh_result;

        //calculate new hidden cells values
        MatrixXd new_hidden_result = temp_output_gate_val * temp_output_tanh_result;
        hidden_values.col(col_index) = new_hidden_result;

        //calculate output layer values(softmax layer)
        MatrixXd new_hidden_bias(hidden_cell_num + 1, 1);
        new_hidden_bias.block(0, 0, hidden_cell_num, 1) = new_hidden_result;
        new_hidden_bias(hidden_cell_num, 0) = 1;    //add bias
        MatrixXd temp_output_val = output_weights * new_hidden_bias;    //(k, h + 1) * (h + 1, 1) = (k, 1)
        //LSTMTool::calculateSoftMax(temp_output_val);
        output_act.col(col_index) = temp_output_val;

    }//end 'for'
    return;
}//end 'forwardPass'

//calculate SoftMax for the output layer
void PointwiseLSTM::calculateSoftMax(MatrixXd output_values)
{
    /*
     * note : this function should be invoked after forwardPass()
     * argument : output_values -- the output layer values needed to calculate SoftMax (K * T)
     * function : calculate SoftMax for the output layer
     */
    for (int t = 0; t < output_values.cols(); ++t)
    {
        double sum = 0;
        for (int row_num = 0; row_num < output_values.rows(); ++row_num)
        {
            output_values(row_num, t) = exp(output_values(row_num, t));
            sum += output_values(row_num, t);
        }
        for (int row_num = 0; row_num < output_values.rows(); ++row_num)
        {
            output_values(row_num, t) = output_values(row_num, t) / sum;
        }
    }
    
    output_act = output_values;
    return;
}

//calculate errors back through the network
void PointwiseLSTM::backwardPass(MatrixXd &label)
{
    /*
     * note : this function should be invoked after forwardPass() and calculateSoftMax()
     * argument : label -- the labels of input datas (K * T)
     * function : calculate errors back through the network
     */
    
    int sequence_length = label.cols();    //sequence length

    //initialize memory matrixes
    error_output = MatrixXd::Zero(output_num, sequence_length);
    error_hidden = MatrixXd::Zero(hidden_cell_num, sequence_length);
    error_input_gate = MatrixXd::Zero(1, sequence_length);
    error_output_gate = MatrixXd::Zero(1, sequence_length);
    error_forget_gate = MatrixXd::Zero(state_cell_num, sequence_length);
    error_input_tanh = MatrixXd::Zero(state_cell_num, sequence_length);
    error_state = MatrixXd::Zero(state_cell_num, sequence_length);
    error_output_tanh = MatrixXd::Zero(hidden_cell_num, sequence_length);

    for (int col_num = sequence_length - 1; col_num >= 0; --col_num)
    {
        //calculate the error of output layer (softmax layer)
        for (int row_num = 0; row_num < output_num; ++row_num)
        {
            if (label(row_num, col_num) == 1)
            {
                error_output(row_num, col_num) = output_act(row_num, col_num) - 1;
            }
            else
            {
                error_output(row_num, col_num) = output_act(row_num, col_num);
            }
        }

        /*
         * calculate hidden cell errors
         */ 
        for (int row_num = 0; row_num < hidden_cell_num; ++row_num)
        {
            //Part 1 : calculate errors from current output layer
            error_hidden(row_num, col_num) = (output_weights.col(row_num).transpose() * error_output.col(col_num))(0, 0);

            if (col_num != sequence_length - 1)
            {
                //Part 2 : calculate errors from next step input_gate 
                error_hidden(row_num, col_num) += error_input_gate(0, col_num + 1) * hidden_input_gate_weights(row_num, 0);
                //Part 3 : calculate errors from next step output_gate 
                error_hidden(row_num, col_num) += error_output_gate(0, col_num + 1) * hidden_output_gate_weights(row_num, 0);
                //Part 4 : calculate errors from next step forget_gate 
                error_hidden(row_num, col_num) += error_forget_gate.col(col_num + 1).transpose() * hidden_forget_gate_weights.col(row_num);
                //Part 5 : calculate errors from next step input_tanh layer
                error_hidden(row_num, col_num) += (error_input_tanh.col(col_num + 1).transpose() * hidden_input_tanh_weights.col(row_num))(0, 0);
            }
        }
        
        //calculate output gate errors
        error_output_gate(0, col_num) = output_gate_act(0, col_num) * (1 - output_gate_act(0, col_num));
        error_output_gate(0, col_num) *= (error_hidden.col(col_num).transpose() * output_tanh_act.col(col_num))(0, 0);

        //calculate output tanh layer errors
        for (int row_num = 0; row_num < hidden_cell_num; ++row_num)
        {
            error_output_tanh(row_num, col_num) = output_gate_act(0, col_num);
            error_output_tanh(row_num, col_num) *= (1 - pow(output_tanh_act(row_num, col_num), 2));
            error_output_tanh(row_num, col_num) *= error_hidden(row_num, col_num);
        }

        /*
         * calculate state layer errors
         */
        for (int row_num = 0; row_num < state_cell_num; ++row_num)
        {
            //Part 1 : calculate errors from current output tanh layer
            error_state(row_num, col_num) = (error_output_tanh.col(col_num).transpose() * output_tanh_weights.col(row_num))(0, 0);
            
            //Part 2 : calculate errors from current output gate
            error_state(row_num, col_num) += error_output_gate(0, col_num) * state_output_gate_weights(row_num, 0);
            
            //next node
            if (col_num != sequence_length - 1)
            {
                //Part 3 : calculate errors from next forget gate
                error_state(row_num, col_num) += error_forget_gate.col(col_num + 1).transpose() * state_forget_gate_weights.col(row_num);
                //Part 4 : calculate errors from next input gate
                error_state(row_num, col_num) += error_input_gate(0, col_num + 1) * state_input_gate_weights(row_num, 0);
                //Part 5 : calculate errors from next state cell layer
                error_state(row_num, col_num) += error_state(row_num, col_num + 1) * forget_gate_act(row_num, col_num + 1);
            }
        }

        //calculate forget gate errors
        if (col_num > 0)
        {
            error_forget_gate.col(col_num) = forget_gate_act.col(col_num).array() * (1 - forget_gate_act.col(col_num).array()) \
                                           * state_values.col(col_num - 1).array() * error_state.col(col_num).array();
        }

        //calculate input gate errors 
        error_input_gate(0, col_num) = input_gate_act(0, col_num) * (1 - input_gate_act(0, col_num));
        error_input_gate(0, col_num) *= (error_state.col(col_num).transpose() * input_tanh_act.col(col_num))(0, 0);

        //calculate input tanh layer errors
        for (int row_num = 0; row_num < state_cell_num; ++row_num)
        {
            error_input_tanh(row_num, col_num) = 1 - pow(input_tanh_act(row_num, col_num), 2);
            error_input_tanh(row_num, col_num) *= input_gate_act(0, col_num);
            error_input_tanh(row_num, col_num) *= error_state(row_num, col_num);
        }

    }//end 'for'
    return;
}//end 'backwardPass()'

//calculate the error of geiven input
double PointwiseLSTM::calculateError(MatrixXd &label)
{
    /*
     * note : this function should be invoked after 'forwardPass()'
     * argument : label : the label of input data (K * T)
     * return : the error of geiven input data
     */

    double error = 0;
    for (int col_num = 0; col_num < label.cols(); ++col_num)
    {
        for (int row_num = 0; row_num < label.rows(); ++row_num)
        {
            if (label(row_num, col_num) == 1)
            {
                error += -log(output_act(row_num, col_num));
                break;
            }
        }
    }
    return error;
}//end 'calculateError()'

//check the derivative is calculated correct or not
void PointwiseLSTM::checkGradient()
{
    /*
     * note : check the derivative of oupout tanh layer's weights
     * note : this function is not integrate but test effective
     */

    //real weight
    MatrixXd true_weight = output_tanh_weights;

    double epsilon = 1e-4;    //EPSILON

    //initialize a random input data and label
    int sequence_length = 100;
    MatrixXd input_data = MatrixXd::Random(input_cell_num, sequence_length);
    MatrixXd label = MatrixXd::Zero(output_num, sequence_length);
    srand((unsigned int)time(NULL));
    for (int col_num = 0; col_num < sequence_length; ++col_num)
    {
        int true_row = rand() % output_num;
        label(true_row, col_num) = 1;
    }

    //calculate derivative with our algorithm
    forwardPass(input_data);
    calculateSoftMax(output_act);
    backwardPass(label);

    double cal_value = (state_values.row(0) * error_output_tanh.row(0).transpose())(0, 0);

    //calculate derivative with approximation
    double sim_value = 0;
    output_tanh_weights(0, 0) += epsilon;
    forwardPass(input_data);
    calculateSoftMax(output_act);

    sim_value = calculateError(label);
    output_tanh_weights(0, 0) -= 2 * epsilon;
    forwardPass(input_data);
    calculateSoftMax(output_act);

    sim_value -= calculateError(label);
    sim_value /= 2 * epsilon;
    
    cout << "cal_value  " << cal_value << endl;
    cout << "sim_value  " << sim_value << endl;

    //restore the real weight
    output_tanh_weights(0, 0) += epsilon;
    
    return;
}

//update network weights with one sentence
void PointwiseLSTM::updateWeightsWithOneSentence(
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
                                       )
{
    /*
     * note : this function is designed for Bidirectional Networks, it is terrible, so if don't use Bi-LSTM, please neglect this fuction
     * argument : now_input_maxtrix -- current input data (I * T)
     *            now_label_maxtrix -- current label data (K * T)
     *            learning_rate -- initial learning rate
     *            momentum_rate -- the parameter of momentum
     *            output_values -- 
     *            other matrixes -- record last derivative for momentum
     */

        int sequence_length = now_input_maxtrix.cols();    //the length of current sequence

        //calculate zhe momentum term
        state_input_gate_weights_derivative *= momentum_rate;
        hidden_input_gate_weights_derivative *= momentum_rate;
        input_input_gate_weights_derivative *= momentum_rate;
        state_output_gate_weights_derivative *= momentum_rate;
        hidden_output_gate_weights_derivative *= momentum_rate;
        input_output_gate_weights_derivative *= momentum_rate;
        state_forget_gate_weights_derivative *= momentum_rate;
        hidden_forget_gate_weights_derivative *= momentum_rate;
        input_forget_gate_weights_derivative *= momentum_rate;

        input_input_tanh_weights_derivative *= momentum_rate;
        hidden_input_tanh_weights_derivative *= momentum_rate;
        output_tanh_weights_derivative *= momentum_rate;
        output_weights_derivative *= momentum_rate;

        //calculate derivative with BPTT
        for (int t = 0; t < sequence_length; ++t)
        {
            //define input, state and hidden vector
            MatrixXd input_now(input_cell_num + 1, 1);
            MatrixXd hidden_previous(hidden_cell_num + 1, 1);
            MatrixXd hidden_now(hidden_cell_num + 1, 1);
            MatrixXd state_previous(state_cell_num + 1, 1);
            MatrixXd state_now(state_cell_num + 1, 1);

            //initialize input, state and hidden vector and add bias for them
            input_now.block(0, 0, input_cell_num, 1) = now_input_maxtrix.col(t);
            input_now(input_cell_num, 0) = 1;
            hidden_now.block(0, 0, hidden_cell_num, 1) = hidden_values.col(t);
            hidden_now(hidden_cell_num, 0) = 1;
            state_now.block(0, 0, state_cell_num, 1) = state_values.col(t);
            state_now(state_cell_num, 0) = 1;
            
            if (t > 0)
            {
                hidden_previous.block(0, 0, hidden_cell_num, 1) = hidden_values.col(t - 1);
                state_previous.block(0, 0, state_cell_num, 1) = state_values.col(t - 1);
            }
            else
            {
                hidden_previous.block(0, 0, hidden_cell_num, 1) = MatrixXd::Zero(hidden_cell_num, 1);
                state_previous.block(0, 0, state_cell_num, 1) = MatrixXd::Zero(state_cell_num, 1);
            }
            
            hidden_previous(hidden_cell_num, 0) = 1;
            state_previous(state_cell_num, 0) = 1;

            //updata forget gate weights derivative (for input, hidden and state layer)
            state_forget_gate_weights_derivative += error_forget_gate.col(t) * state_previous.transpose() * learning_rate;
            hidden_forget_gate_weights_derivative += error_forget_gate.col(t) * hidden_previous.transpose() * learning_rate;
            input_forget_gate_weights_derivative +=  error_forget_gate.col(t) * input_now.transpose() * learning_rate;

            //updata input gate weights derivative (for input, hidden and state layer)
            state_input_gate_weights_derivative += error_input_gate(0, t) * state_previous * learning_rate;
            hidden_input_gate_weights_derivative += error_input_gate(0, t) * hidden_previous * learning_rate;
            input_input_gate_weights_derivative += error_input_gate(0, t) * input_now * learning_rate;

            //updata output gate weights derivative (for input, hidden and state layer)
            hidden_output_gate_weights_derivative += error_output_gate(0, t) * hidden_previous * learning_rate;
            state_output_gate_weights_derivative += error_output_gate(0, t) * state_now * learning_rate;
            input_output_gate_weights_derivative += error_output_gate(0, t) * input_now * learning_rate;

            //updata input tanh layer weights derivative w.r.t. input layer 
            input_input_tanh_weights_derivative += error_input_tanh.col(t) * input_now.transpose() * learning_rate;

            //updata input tanh layer weights derivative w.r.t. hidden layer 
            hidden_input_tanh_weights_derivative += error_input_tanh.col(t) * hidden_previous.transpose() * learning_rate;
            
            //updata output tanh layer weights derivative
            output_tanh_weights_derivative += error_output_tanh.col(t) * state_now.transpose() * learning_rate;

            //updata output layer weights derivative (softmax layer)
            output_weights_derivative += error_output.col(t) * hidden_now.transpose() * learning_rate;
        }//end 'for' of sequence

        //updata weights
        state_input_gate_weights -= state_input_gate_weights_derivative;
        hidden_input_gate_weights -= hidden_input_gate_weights_derivative;
        input_input_gate_weights -= input_input_gate_weights_derivative;
        state_output_gate_weights -= state_output_gate_weights_derivative;
        hidden_output_gate_weights -= hidden_output_gate_weights_derivative;
        input_output_gate_weights -= input_output_gate_weights_derivative;
        state_forget_gate_weights -= state_forget_gate_weights_derivative;
        hidden_forget_gate_weights -= hidden_forget_gate_weights_derivative;
        input_forget_gate_weights -= input_forget_gate_weights_derivative;

        input_input_tanh_weights -= input_input_tanh_weights_derivative;
        hidden_input_tanh_weights -= hidden_input_tanh_weights_derivative;
        output_tanh_weights -= output_tanh_weights_derivative;
        output_weights -= output_weights_derivative;
        
        double ave_error = calculateError(now_label_maxtrix) / sequence_length;
        cout << ave_error << endl;
    
    return;
}


//SGD with a momentum
void PointwiseLSTM::stochasticGradientDescent(vector<MatrixXd*> input_datas, vector<MatrixXd*> input_labels, double learning_rate, double momentum_rate = 0)
{
    /*
     * note : the stopping criteria is 
     * argument : input_datas -- a vector of input data, every element is a I * T matrix
     *            input_labels -- a vector of input labels, every element is a K * T matrix
     *            learning_rate -- initial learning rate
     *            momentum_rate -- the parameter of momentum
     */
    
    //matrixes record previous derivative
    MatrixXd state_input_gate_weights_derivative = MatrixXd::Zero(state_cell_num + 1, 1);
    MatrixXd hidden_input_gate_weights_derivative = MatrixXd::Zero(hidden_cell_num + 1, 1);
    MatrixXd input_input_gate_weights_derivative = MatrixXd::Zero(input_cell_num + 1, 1);
    MatrixXd state_output_gate_weights_derivative = MatrixXd::Zero(state_cell_num + 1, 1);
    MatrixXd hidden_output_gate_weights_derivative = MatrixXd::Zero(hidden_cell_num + 1, 1);
    MatrixXd input_output_gate_weights_derivative = MatrixXd::Zero(input_cell_num + 1, 1);
    MatrixXd state_forget_gate_weights_derivative = MatrixXd::Zero(state_cell_num, state_cell_num + 1);
    MatrixXd hidden_forget_gate_weights_derivative = MatrixXd::Zero(state_cell_num, hidden_cell_num + 1);
    MatrixXd input_forget_gate_weights_derivative = MatrixXd::Zero(state_cell_num, input_cell_num + 1);

    MatrixXd input_input_tanh_weights_derivative = MatrixXd::Zero(state_cell_num, input_cell_num + 1);
    MatrixXd hidden_input_tanh_weights_derivative = MatrixXd::Zero(state_cell_num, hidden_cell_num + 1);
    MatrixXd output_tanh_weights_derivative = MatrixXd::Zero(hidden_cell_num, state_cell_num + 1);
    MatrixXd output_weights_derivative = MatrixXd::Zero(output_num, hidden_cell_num + 1);
    
    //the number of current input
    int input_count = 0;
    //total number of input datas
    int input_num = input_datas.size();

    for (int i = 0; i < 1; ++i)
    {

    //randomising the order of input datas and labels
    LSTMTool::disturbOrder(input_datas, input_labels);

    for (int input_index = 0; input_index < input_num; ++input_index)
    {
        ++input_count;
        
        MatrixXd now_input_maxtrix = *(input_datas[input_index]);    //current input data
        MatrixXd now_label_maxtrix = *(input_labels[input_index]);    //current input label
        
        forwardPass(now_input_maxtrix);
        calculateSoftMax(output_act);
        backwardPass(now_label_maxtrix);

        int sequence_length = now_input_maxtrix.cols();    //the length of current sequence

        //calculate zhe momentum term
        state_input_gate_weights_derivative *= momentum_rate;
        hidden_input_gate_weights_derivative *= momentum_rate;
        input_input_gate_weights_derivative *= momentum_rate;
        state_output_gate_weights_derivative *= momentum_rate;
        hidden_output_gate_weights_derivative *= momentum_rate;
        input_output_gate_weights_derivative *= momentum_rate;
        state_forget_gate_weights_derivative *= momentum_rate;
        hidden_forget_gate_weights_derivative *= momentum_rate;
        input_forget_gate_weights_derivative *= momentum_rate;

        input_input_tanh_weights_derivative *= momentum_rate;
        hidden_input_tanh_weights_derivative *= momentum_rate;
        output_tanh_weights_derivative *= momentum_rate;
        output_weights_derivative *= momentum_rate;

        //calculate derivative with BPTT
        for (int t = 0; t < sequence_length; ++t)
        {
            //define input, state and hidden vector
            MatrixXd input_now(input_cell_num + 1, 1);
            MatrixXd hidden_previous(hidden_cell_num + 1, 1);
            MatrixXd hidden_now(hidden_cell_num + 1, 1);
            MatrixXd state_previous(state_cell_num + 1, 1);
            MatrixXd state_now(state_cell_num + 1, 1);

            //initialize input, state and hidden vector and add bias for them
            input_now.block(0, 0, input_cell_num, 1) = now_input_maxtrix.col(t);
            input_now(input_cell_num, 0) = 1;
            hidden_now.block(0, 0, hidden_cell_num, 1) = hidden_values.col(t);
            hidden_now(hidden_cell_num, 0) = 1;
            state_now.block(0, 0, state_cell_num, 1) = state_values.col(t);
            state_now(state_cell_num, 0) = 1;
            
            if (t > 0)
            {
                hidden_previous.block(0, 0, hidden_cell_num, 1) = hidden_values.col(t - 1);
                state_previous.block(0, 0, state_cell_num, 1) = state_values.col(t - 1);
            }
            else
            {
                hidden_previous.block(0, 0, hidden_cell_num, 1) = MatrixXd::Zero(hidden_cell_num, 1);
                state_previous.block(0, 0, state_cell_num, 1) = MatrixXd::Zero(state_cell_num, 1);
            }
            
            hidden_previous(hidden_cell_num, 0) = 1;
            state_previous(state_cell_num, 0) = 1;

            //updata forget gate weights derivative (for input, hidden and state layer)
            state_forget_gate_weights_derivative += error_forget_gate.col(t) * state_previous.transpose() * learning_rate;
            hidden_forget_gate_weights_derivative += error_forget_gate.col(t) * hidden_previous.transpose() * learning_rate;
            input_forget_gate_weights_derivative +=  error_forget_gate.col(t) * input_now.transpose() * learning_rate;

            //updata input gate weights derivative (for input, hidden and state layer)
            state_input_gate_weights_derivative += error_input_gate(0, t) * state_previous * learning_rate;
            hidden_input_gate_weights_derivative += error_input_gate(0, t) * hidden_previous * learning_rate;
            input_input_gate_weights_derivative += error_input_gate(0, t) * input_now * learning_rate;

            //updata output gate weights derivative (for input, hidden and state layer)
            hidden_output_gate_weights_derivative += error_output_gate(0, t) * hidden_previous * learning_rate;
            state_output_gate_weights_derivative += error_output_gate(0, t) * state_now * learning_rate;
            input_output_gate_weights_derivative += error_output_gate(0, t) * input_now * learning_rate;

            //updata input tanh layer weights derivative w.r.t. input layer 
            input_input_tanh_weights_derivative += error_input_tanh.col(t) * input_now.transpose() * learning_rate;

            //updata input tanh layer weights derivative w.r.t. hidden layer 
            hidden_input_tanh_weights_derivative += error_input_tanh.col(t) * hidden_previous.transpose() * learning_rate;
            
            //updata output tanh layer weights derivative
            output_tanh_weights_derivative += error_output_tanh.col(t) * state_now.transpose() * learning_rate;

            //updata output layer weights derivative (softmax layer)
            output_weights_derivative += error_output.col(t) * hidden_now.transpose() * learning_rate;
        }//end 'for' of sequence

        //updata weights
        state_input_gate_weights -= state_input_gate_weights_derivative;
        hidden_input_gate_weights -= hidden_input_gate_weights_derivative;
        input_input_gate_weights -= input_input_gate_weights_derivative;
        state_output_gate_weights -= state_output_gate_weights_derivative;
        hidden_output_gate_weights -= hidden_output_gate_weights_derivative;
        input_output_gate_weights -= input_output_gate_weights_derivative;
        state_forget_gate_weights -= state_forget_gate_weights_derivative;
        hidden_forget_gate_weights -= hidden_forget_gate_weights_derivative;
        input_forget_gate_weights -= input_forget_gate_weights_derivative;

        input_input_tanh_weights -= input_input_tanh_weights_derivative;
        hidden_input_tanh_weights -= hidden_input_tanh_weights_derivative;
        output_tanh_weights -= output_tanh_weights_derivative;
        output_weights -= output_weights_derivative;
        
        double ave_error = calculateError(now_label_maxtrix) / sequence_length;
        cout << ave_error << endl;

    }//end 'for' of input datas
    
    }//end 'for' of pass num
    return;
}

//predict the output of the geiven input datas
vector<MatrixXd*> PointwiseLSTM::predict(vector<MatrixXd*> input_datas)
{
    /*
     * argument : input_datas -- the data waiting to be predict, every element is a matrix pointer (I * T)
     * return : predict_labels -- the label of input datas predicted by LSTM, every element is a matrix pointer (K * T)
     */

    vector<MatrixXd*> predict_labels;    //vector to store the predict labels
    int input_num = input_datas.size();    //the number of input datas

    //for every input predict the label
    for (int i = 0; i < input_num; ++i)
    {
        MatrixXd temp_input_data = *(input_datas[i]);
        int sequence_length = temp_input_data.cols();

        forwardPass(temp_input_data);    //forward pass two calculte the value of every node
        calculateSoftMax(output_act);

        MatrixXd *temp_predict_label = new MatrixXd(output_num, sequence_length);
        *temp_predict_label = MatrixXd::Zero(output_num, sequence_length);

        for (int col_num = 0; col_num < sequence_length; ++col_num)
        {
            int temp_row = 0;
            int temp_col = 0;
            int *p_temp_row = &temp_row;
            int *p_temp_col = &temp_col;
            output_act.col(col_num).maxCoeff(p_temp_row, p_temp_col);
            (*temp_predict_label)(*p_temp_row, col_num) = 1;
        }

        predict_labels.push_back(temp_predict_label);
    }

    return predict_labels;
}

//save the model into a file
void PointwiseLSTM::saveModel(string file_name)
{
    /*
     * argument : file_name -- the file to store the model
     * note : the first line of the file will store various nodes number with the Declaration order.
     *        the weights of model will be then writen into the file with the Declaration order.
     */
    
    ofstream of_model(file_name.c_str());

    char block = ' ';
    of_model << input_cell_num << block << hidden_cell_num << block << state_cell_num << block << output_num << endl;
    of_model << state_input_gate_weights << endl;
    of_model << hidden_input_gate_weights << endl;
    of_model << input_input_gate_weights << endl;
    of_model << state_output_gate_weights << endl;
    of_model << hidden_output_gate_weights << endl;
    of_model << input_output_gate_weights << endl;
    of_model << state_forget_gate_weights << endl;
    of_model << hidden_forget_gate_weights << endl;
    of_model << input_forget_gate_weights << endl;
    of_model << input_input_tanh_weights << endl;
    of_model << hidden_input_tanh_weights << endl;
    of_model << output_tanh_weights << endl;
    of_model << output_weights << endl;
    
    of_model.close();
    return;
}

//load model from geiven file
void PointwiseLSTM::loadModel(string file_name)
{
    /*
     * argument : file_name -- the file saves the weights of model
     * note : the first line of the file will store various nodes number with the Declaration order.
     *        the weights of model will be then writen into the file with the Declaration order.
     */
    
    ifstream if_model(file_name.c_str());

    //load various layer cell numner with the declaration order
    if_model >> input_cell_num >> hidden_cell_num >> state_cell_num >> output_num;

    //resize the weights matrixes
    state_input_gate_weights.resize(state_cell_num + 1, 1);
    hidden_input_gate_weights.resize(hidden_cell_num + 1, 1);
    input_input_gate_weights.resize(input_cell_num + 1, 1);
    state_output_gate_weights.resize(state_cell_num + 1, 1);
    hidden_output_gate_weights.resize(hidden_cell_num + 1, 1);
    input_output_gate_weights.resize(input_cell_num + 1, 1);
    state_forget_gate_weights.resize(state_cell_num, state_cell_num + 1);
    hidden_forget_gate_weights.resize(state_cell_num, hidden_cell_num + 1);
    input_forget_gate_weights.resize(state_cell_num, input_cell_num + 1);
    
    input_input_tanh_weights.resize(state_cell_num, input_cell_num + 1);
    hidden_input_tanh_weights.resize(state_cell_num, hidden_cell_num + 1);
    output_tanh_weights.resize(hidden_cell_num, state_cell_num + 1);
    output_weights.resize(output_num, hidden_cell_num + 1);
    
    //load state_input_gate_weights
    for (int i = 0; i < state_cell_num + 1; ++i)
    {
        if_model >> state_input_gate_weights(i, 0);
    }
    
    //load hidden_input_gate_weights
    for (int i = 0; i < hidden_cell_num + 1; ++i)
    {
        if_model >> hidden_input_gate_weights(i, 0);
    }

    //load input_input_gate_weights
    for (int i = 0; i < input_cell_num + 1; ++i)
    {
        if_model >> input_input_gate_weights(i, 0);
    }

    //load state_output_gate_weights
    for (int i = 0; i < state_cell_num + 1; ++i)
    {
        if_model >> state_output_gate_weights(i, 0);
    }

    //load hidden_output_gate_weights
    for (int i = 0; i < hidden_cell_num + 1; ++i)
    {
        if_model >> hidden_output_gate_weights(i, 0);
    }

    //load input_output_gate_weights
    for (int i = 0; i < input_cell_num + 1; ++i)
    {
        if_model >> input_output_gate_weights(i, 0);
    }

    //load state_forget_gate_weights
    for (int row_index = 0; row_index < state_cell_num; ++row_index)
    {
        for (int col_index = 0; col_index < state_cell_num + 1; ++col_index)
        {
            if_model >> state_forget_gate_weights(row_index, col_index);
        }
    }

    //load hidden_forget_gate_weights
    for (int row_index = 0; row_index < state_cell_num; ++row_index)
    {
        for (int col_index = 0; col_index < hidden_cell_num + 1; ++col_index)
        {
            if_model >> hidden_forget_gate_weights(row_index, col_index);
        }
    }

    //load input_forget_gate_weights
    for (int row_index =0; row_index < state_cell_num; ++row_index)
    {
        for (int col_index = 0; col_index < input_cell_num + 1; ++col_index)
        {
            if_model >> input_forget_gate_weights(row_index, col_index);
        }
    }

    //load input_input_tanh_weights
    for (int row_index = 0; row_index < state_cell_num; ++row_index)
    {
        for (int col_index = 0; col_index < input_cell_num + 1; ++col_index)
        {
            if_model >> input_input_tanh_weights(row_index, col_index);
        }
    }

    //load hidden_input_tanh_weights
    for (int row_index = 0; row_index < state_cell_num; ++row_index)
    {
        for (int col_index = 0; col_index < hidden_cell_num + 1; ++col_index)
        {
            if_model >> hidden_input_tanh_weights(row_index, col_index);
        }
    }

    //load output_tanh_weights
    for (int row_index = 0; row_index < hidden_cell_num; ++row_index)
    {
        for (int col_index = 0; col_index < state_cell_num + 1; ++col_index)
        {
            if_model >> output_tanh_weights(row_index, col_index);
        }
    }

    //load output_weights
    for (int row_index = 0; row_index < output_num; ++row_index)
    {
        for (int col_index = 0; col_index < hidden_cell_num + 1; ++ col_index)
        {
            if_model >> output_weights(row_index, col_index);
        }
    }

    if_model.close();

    return;
}

//get the matrix of output_act
MatrixXd PointwiseLSTM::getOutputValue()
{
    return output_act;
}

//get the value of input_cell_num
int PointwiseLSTM::getInputCellNum()
{
    return input_cell_num;
}

//get the value of state_cell_num
int PointwiseLSTM::getStateCellNum()
{
    return state_cell_num;
}

//get the value of hidden_cell_num
int PointwiseLSTM::getHiddenCellNum()
{
    return hidden_cell_num;
}

//get the value of output_num
int PointwiseLSTM::getOutputNum()
{
    return output_num;
}

/*
int main()
{
    PointwiseLSTM lstm(100, 80, 120, 3);
    lstm.checkGradient();
    /*
    MatrixXd input_data = MatrixXd::Random(100, 10);
    MatrixXd label = MatrixXd::Zero(3, 10);
    srand((unsigned int)time(NULL));
    for (int col_num = 0; col_num < 10; ++col_num)
    {
        int true_row = rand() % 3;
        label(true_row, col_num);
    }
    lstm.forwardPass(input_data);
    lstm.backwardPass(label);


    lstm.loadModel("__Model");
    lstm.saveModel("__Model2");
    
    return 0;
}
*/
