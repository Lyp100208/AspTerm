/*************************************************************************
    > File Name: lstm.cpp
    > Author: Wang Junqi
    > Mail: 351868656@qq.com 
    > Created Time: 2015年12月29日 星期二 10时04分29秒
 ************************************************************************/

#include<iostream>
#include<cmath>

#include"../include/lstm.h"
#include"../include/tool.h"

using namespace std;

//initialization
LSTM::LSTM(int input_cell_num, int hidden_cell_num, int state_cell_num, int output_num)
{
    this -> input_cell_num = input_cell_num;
    this -> hidden_cell_num = hidden_cell_num;
    this -> state_cell_num = state_cell_num;
    this -> output_num = output_num;
    
    //initialize weights randomly from [-0.1, 0.1]
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

    return;
}

//forward pass geiven the input data (matrix)
void LSTM::forwardPass(MatrixXd temp_input_matrix)
{
    /*
     * input_matrix : input sequence (not added bias) (I * T)
     */
    
    int sequence_length = temp_input_matrix.cols();    //sequence length

    //initialize memory matrix with zero
    state_values = MatrixXd::Zero(state_cell_num, sequence_length);
    hidden_values = MatrixXd::Zero(hidden_cell_num, sequence_length);
    input_gate_act = MatrixXd::Zero(1, sequence_length);
    output_gate_act = MatrixXd::Zero(1, sequence_length);
    forget_gate_act = MatrixXd::Zero(1, sequence_length);
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
    	temp_state_result = (temp_state_trans * state_forget_gate_weights)(0, 0);
    	temp_hidden_result = (temp_hidden_trans * hidden_forget_gate_weights)(0, 0);
    	temp_input_result = (temp_input_trans * input_forget_gate_weights)(0, 0);
    	double temp_forget_gate_val = Tool::sigmoid(temp_state_result + temp_hidden_result + temp_input_result);
    	forget_gate_act(0, col_index) = temp_forget_gate_val;
    	
    	//calculate the activation of input gate
    	temp_state_result = (temp_state_trans * state_input_gate_weights)(0, 0);
    	temp_hidden_result = (temp_hidden_trans * hidden_input_gate_weights)(0, 0);
    	temp_input_result = (temp_input_trans * input_input_gate_weights)(0, 0);
    	double temp_input_gate_val = Tool::sigmoid(temp_state_result + temp_hidden_result + temp_input_result);
    	input_gate_act(0, col_index) = temp_input_gate_val;

    	//calculate input tanh layer activation
    	MatrixXd input_input_tanh_result = input_input_tanh_weights * temp_input;
        MatrixXd hidden_input_tanh_result = hidden_input_tanh_weights * temp_hidden_trans.transpose();
        MatrixXd temp_input_tanh_result = input_input_tanh_result + hidden_input_tanh_result;
        Tool::tanhXforMatrix(temp_input_tanh_result);
        input_tanh_act.col(col_index) = temp_input_tanh_result;
        
        //calculate new state cells values
        MatrixXd new_state_result = temp_forget_gate_val * temp_state_trans.transpose().block(0, 0, state_cell_num, 1) + temp_input_gate_val * temp_input_tanh_result;
        state_values.col(col_index) = new_state_result;
        MatrixXd new_state_trans(1, state_cell_num + 1);
        new_state_trans.block(0, 0, 1, state_cell_num) = new_state_result.transpose();
        new_state_trans(0, state_cell_num) = 1;    //add bias

    	//calculate the activation of output gate
        temp_state_result = (new_state_trans * state_output_gate_weights)(0, 0);
        temp_hidden_result = (temp_hidden_trans * hidden_output_gate_weights)(0, 0);
        temp_input_result = (temp_input_trans * input_output_gate_weights)(0, 0);
        double temp_output_gate_val = Tool::sigmoid(temp_state_result + temp_hidden_result + temp_input_result);
        output_gate_act(0, col_index) = temp_output_gate_val;

        //calculate output tanh layer
        MatrixXd temp_output_tanh_result = output_tanh_weights * new_state_trans.transpose();
        Tool::tanhXforMatrix(temp_output_tanh_result);
        output_tanh_act.col(col_index) = temp_output_tanh_result;

        //calculate new hidden cells values
        MatrixXd new_hidden_result = temp_output_gate_val * temp_output_tanh_result;
        hidden_values.col(col_index) = new_hidden_result;

        //calculate output layer values(softmax layer)
        MatrixXd new_hidden_bias(hidden_cell_num + 1, 1);
        new_hidden_bias.block(0, 0, hidden_cell_num, 1) = new_hidden_result;
        new_hidden_bias(hidden_cell_num, 0) = 1;    //add bias
        MatrixXd temp_output_val = output_weights * new_hidden_bias;    //(k, h + 1) * (h + 1, 1) = (k, 1)
        Tool::calculateSoftMax(temp_output_val);
        output_act.col(col_index) = temp_output_val;

    }//end 'for'
    return;
}//end 'forwardPass'

//calculate errors back through the network
void LSTM::backwardPass(MatrixXd &label)
{
    /*
     * note : this function should be invoked after forwardPass()
     * argument : label -- the labels of input datas (K * T)
     * function : calculate errors back through the network
     */
    
    int sequence_length = label.cols();    //sequence length

    //initialize memory matrixes
    error_output = MatrixXd::Zero(output_num, sequence_length);
    error_hidden = MatrixXd::Zero(hidden_cell_num, sequence_length);
    error_input_gate = MatrixXd::Zero(1, sequence_length);
    error_output_gate = MatrixXd::Zero(1, sequence_length);
    error_forget_gate = MatrixXd::Zero(1, sequence_length);
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
                error_hidden(row_num, col_num) += error_forget_gate(0, col_num + 1) * hidden_forget_gate_weights(row_num, 0);
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
                error_state(row_num, col_num) += error_forget_gate(0, col_num + 1) * state_forget_gate_weights(row_num, 0);
                //Part 4 : calculate errors from next input gate
                error_state(row_num, col_num) += error_input_gate(0, col_num + 1) * state_input_gate_weights(row_num, 0);
                //Part 5 : calculate errors from next state cell layer
                error_state(row_num, col_num) += error_state(row_num, col_num + 1) * forget_gate_act(0, col_num + 1);
            }
        }

        //calculate forget gate errors
        if (col_num > 0)
        {
            error_forget_gate(0, col_num) = forget_gate_act(0, col_num) * (1 - forget_gate_act(0, col_num));
            error_forget_gate(0, col_num) *= (state_values.col(col_num - 1).transpose() * error_state.col(col_num))(0, 0);
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
double LSTM::calculateError(MatrixXd &label)
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
void LSTM::checkGradient()
{
    /*
     * note : check the derivative of oupout tanh layer's weights
     * note : this function is not integrate but test effective
     */

    //real weight
    MatrixXd true_weight = output_tanh_weights;

    double epsilon = 1e-7;    //EPSILON

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
    backwardPass(label);
    double cal_value = (state_values.row(0) * error_output_tanh.row(0).transpose())(0, 0);

    //calculate derivative with approximation
    double sim_value = 0;
    output_tanh_weights(0, 0) += epsilon;
    forwardPass(input_data);
    sim_value = calculateError(label);
    output_tanh_weights(0, 0) -= 2 * epsilon;
    forwardPass(input_data);
    sim_value -= calculateError(label);
    sim_value /= 2 * epsilon;
    
    cout << "cal_value  " << cal_value << endl;
    cout << "sim_value  " << sim_value << endl;

    //restore the real weight
    output_tanh_weights(0, 0) += epsilon;
    
    return;
}

//SGD with a momentum
void LSTM::stochasticGradientDescent(vector<MatrixXd*> input_datas, vector<MatrixXd*> input_labels, double learning_rate, double momentum_rate = 0)
{
    /*
     * note : the stopping criteria is 
     * argument : input_datas -- a vector of input data, every element is a I * T matrix
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
    MatrixXd state_forget_gate_weights_derivative = MatrixXd::Zero(state_cell_num + 1, 1);
    MatrixXd hidden_forget_gate_weights_derivative = MatrixXd::Zero(hidden_cell_num + 1, 1);
    MatrixXd input_forget_gate_weights_derivative = MatrixXd::Zero(input_cell_num + 1, 1);

    MatrixXd input_input_tanh_weights_derivative = MatrixXd::Zero(state_cell_num, input_cell_num + 1);
    MatrixXd hidden_input_tanh_weights_derivative = MatrixXd::Zero(state_cell_num, hidden_cell_num + 1);
    MatrixXd output_tanh_weights_derivative = MatrixXd::Zero(hidden_cell_num, state_cell_num + 1);
    MatrixXd output_weights_derivative = MatrixXd::Zero(output_num, hidden_cell_num + 1);
    
    //the number of current input
    int input_count = 0;
    //total number of input datas
    int input_num = input_datas.size();

    for (int input_index = 0; input_index < input_num; ++input_index)
    {
        ++input_count;
        //randomising the order of input datas and labels
        Tool::disturbOrder(input_datas, input_labels);
        
        MatrixXd now_input_maxtrix = *(input_datas[input_index]);    //current input data
        MatrixXd now_label_maxtrix = *(input_labels[input_index]);    //current input label
        
        forwardPass(now_input_maxtrix);
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
            state_forget_gate_weights_derivative += error_forget_gate(0, t) * state_previous * learning_rate;
            hidden_forget_gate_weights_derivative += error_forget_gate(0, t) * hidden_previous * learning_rate;
            input_forget_gate_weights_derivative +=  error_forget_gate(0, t) * input_now * learning_rate;

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
        

    }//end 'for' of input datas
    
    
    return;
}


int main()
{
    LSTM lstm(100, 80, 120, 3);
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
    */
    return 0;
}