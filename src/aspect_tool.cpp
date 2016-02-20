/*************************************************************************
	> File Name: aspect_tool.cpp
	> Author: Wang Junqi
	> Mail: 351868656@qq.com 
	> Created Time: 2016年01月09日 星期六 17时05分09秒
 ************************************************************************/

#include"../include/aspect_tool.h"
#include"../include/word2vec_tool.h"
#include"../include/string_tool.h"
#include"../include/lstm.h"
#include<fstream>

void AspectTool::getInputData(string file_name, vector<MatrixXd*> &input_datas, vector<MatrixXd*> &input_labels)
{

    ifstream input_stream(file_name.c_str());
 
    if (!input_stream)
    {
        cout  << file_name << " dose not exist !" << endl;
        return;
    }
    
    Word2VecTool word_vec_tool("../res/glove.42B.300d.txt", false);
    //word_vec_tool.initWordIndexDict();    
    word_vec_tool.loadWordDict("word_index_dictionary");
    
    string temp_word;

    vector<string> temp_word_list;
    vector<char> temp_label_list;

    //int count = 0;
    while (input_stream >> temp_word)
    {
        if (temp_word == "<end_of_sentence>")
        {
            //construct input matrix of current sequence
            MatrixXd *temp_input_mp = new MatrixXd(word_vec_tool.getVectorLength(), temp_word_list.size());
            for (int t = 0; t < temp_word_list.size(); ++t)
            {
                MatrixXd temp_matrix(word_vec_tool.getVectorLength(), 1);
                word_vec_tool.getWrodVect(temp_word_list[t], temp_matrix);
                temp_input_mp -> col(t) = temp_matrix;
            }

            //construct label matrix of current sequence
            MatrixXd *temp_label_mp = new MatrixXd(3, temp_label_list.size());
            *temp_label_mp = MatrixXd::Zero(3, temp_label_list.size());
            for (int t = 0; t < temp_label_list.size(); ++t)
            {
                if (temp_label_list[t] == 'O')
                {
                    (*temp_label_mp)(0, t) = 1;
                }
                else if (temp_label_list[t] == 'B')
                {
                    (*temp_label_mp)(1, t) = 1;
                }
                else if (temp_label_list[t] == 'I')
                {
                    (*temp_label_mp)(2, t) = 1;
                }
            }

            //insert current input data and label
            input_datas.push_back(temp_input_mp);
            input_labels.push_back(temp_label_mp);

            //clear the list
            temp_word_list.clear();
            temp_label_list.clear();
        }
        else
        {
            temp_word_list.push_back(temp_word);
            char temp_char;
            input_stream >> temp_char;
            temp_label_list.push_back(temp_char);
        }
    }// end 'while'

    return;
}//end 'getInputData'

//calculata f1 score of the predict labels
double AspectTool::calF1Score(const vector<MatrixXd*> &predict_labels, const vector<MatrixXd*> &true_labels)
{
    
    /*
     * argument : predict_labels -- the label predict by Sequence Labeler
     *            true_labels -- the true label 
     * return : f1 score
     */

    double predict_num = 0;
    double true_num = 0;
    double predict_true_num = 0;
    
    for (int input_index = 0; input_index < predict_labels.size(); ++input_index)
    {
        MatrixXd *p_predict_label = predict_labels[input_index];    //current predict label
        MatrixXd *p_true_label = true_labels[input_index];    //current predict label

        int predict_continue_num = 0;    //the num of I after B detected
        int true_continue_num = 0;    //the num of I after B detected

        for (int col_num = 0; col_num < p_predict_label -> cols(); ++col_num)
        {
            
            //check the predict label
            if ((*p_predict_label)(1, col_num) == 1)
            {
                predict_continue_num = 1;
                predict_num += 1;
            }
            else if ((*p_predict_label)(2, col_num) == 1 && predict_continue_num > 0)
            {
                predict_continue_num += 1;
            }
            else if ((*p_predict_label)(0, col_num) == 1)
            {
                if (predict_continue_num > 0 && predict_continue_num == true_continue_num)
                {
                    predict_true_num += 1;
                }
                predict_continue_num = 0;
            }

            //check the true label
            if ((*p_true_label)(1, col_num) == 1)
            {
                true_continue_num = 1;
                true_num += 1;
            }
            else if ((*p_true_label)(2, col_num) == 1 && true_continue_num > 0)
            {
                true_continue_num += 1;
            }
            else if ((*p_true_label)(0, col_num) == 1)
            {
                true_continue_num = 0;
            }

        }// end 'for'

        //calculate zhe aspect term at the end of sectence
        if (predict_continue_num > 0 && predict_continue_num == true_continue_num)
        {
            ++predict_true_num;
        }
    }

    double accuracy_rate = predict_true_num / predict_num;    //calculate accuracy rate
    double call_back_rete = predict_true_num / true_num;    //calculate call back rate
    cout << "a_r : " << accuracy_rate<< endl;
    cout << "c_r : " << call_back_rete << endl; 
    cout << "predict_true_num : " << predict_true_num << endl;
    cout << "predict_num : " << predict_num << endl; 
    cout << "true_num : " << true_num << endl; 

    double f1 = 2 * accuracy_rate * call_back_rete / (accuracy_rate + call_back_rete);    //calculate f1 score : f1 = 2 * a * c / (a + c)

    cout << "f1 : " << f1 << endl;
    return f1;
}

//input from console and output the result predict by the model
void AspectTool::realTest(string model_file)
{
    //initialize word vector tool
    Word2VecTool word_vec_tool("../res/glove.42B.300d.txt", false);
    //word_vec_tool.initWordIndexDict();    
    word_vec_tool.loadWordDict("word_index_dictionary");
    
    cout << "Please input the test sectence : " << endl;
    string sentence;
    getline(cin, sentence);

cout << sentence << endl;
    StringTool st;
    vector<string> word_token = st.tokenize(sentence);
    
    MatrixXd *p_input_data = new MatrixXd(word_vec_tool.getVectorLength(), word_token.size());
    for (int i = 0; i < word_token.size(); ++i)
    {
        MatrixXd temp_matrix(word_vec_tool.getVectorLength(), 1);
        word_vec_tool.getWrodVect(word_token[i], temp_matrix);
        p_input_data -> col(i) = temp_matrix;
    }

    vector<MatrixXd*> input_datas;
    input_datas.push_back(p_input_data);

//cout << *p_input_data << endl;
    LSTM lstm(model_file);

    vector<MatrixXd*> predict_labels = lstm.predict(input_datas);

    MatrixXd temp_label = *(predict_labels[0]);

cout << temp_label << endl;
    string aspect_term = "";
    for (int i = 0; i < temp_label.cols(); ++i)
    {
        if (temp_label(1, i) == 1)
        {
            aspect_term = aspect_term + " " + word_token[i];
        }
        else if (temp_label(2, i) == 1 && aspect_term.size() > 0)
        {
            aspect_term = aspect_term + " " + word_token[i];
        }
        else if (temp_label(0, i) == 1)
        {
            if (aspect_term.size() > 0)
            {
                cout << "Aspect Term : " << aspect_term << endl;
            }
            aspect_term = "";
        }
    }

    if (aspect_term.size() > 0)
    {
        cout << "Aspect Term : " << aspect_term << endl;
    }

    return;
}

int main()
{
    AspectTool asp_tool;
    asp_tool.realTest("320_350_0.001_9_m");
    return 0;
}

