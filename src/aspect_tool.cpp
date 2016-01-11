/*************************************************************************
	> File Name: aspect_tool.cpp
	> Author: Wang Junqi
	> Mail: 351868656@qq.com 
	> Created Time: 2016年01月09日 星期六 17时05分09秒
 ************************************************************************/

#include"../include/aspect_tool.h"
#include"../include/word2vec_tool.h"
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
    word_vec_tool.initWordIndexDict();    
    
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


int main()
{
    AspectTool asp_tool;
    vector<MatrixXd*> input_datas;
    vector<MatrixXd*> input_labels;
    asp_tool.getInputData("../res/raw_train", input_datas, input_labels);
    return 0;
}
