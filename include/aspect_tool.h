/*************************************************************************
	> File Name: aspect_tool.h
	> Author: Wang Junqi
	> Mail: 351868656@qq.com 
	> Created Time: 2016年01月09日 星期六 15时49分45秒
 ************************************************************************/

#ifndef ASPECT_TOOL_H
#define ASPECT_TOOL_H

#include<iostream>
#include<vector>
#include"../lib/Eigen/Dense"

using namespace std;
using namespace Eigen;

class AspectTool
{
    
public:

    void getInputData(string file_name, vector<MatrixXd*> &input_datas, vector<MatrixXd*> &input_labels);    //get Input data and Input label from geiven file
    double calF1Score(const vector<MatrixXd*> &predict_labels, const vector<MatrixXd*> &true_labels);    //calculate f1 score of the predict labels
    void realTest(string model_file);    //input from console and output the result predict by the model
};

#endif
