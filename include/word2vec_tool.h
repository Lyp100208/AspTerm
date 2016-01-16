/*************************************************************************
	> File Name: word2vec_tool.h
	> Author: Wang Junqi
	> Mail: 351868656@qq.com 
	> Created Time: 2016年01月08日 星期五 21时10分14秒
 ************************************************************************/

#ifndef WORD2VEC_TOOL_H
#define WORD2VEC_TOOL_H

#include<iostream>
#include<map>
#include"../lib/Eigen/Dense"

using namespace std;
using namespace Eigen;

class Word2VecTool
{

private:

    map<string, long long> word_index_dict;    //word vector dict recording the word_offset in the word vector file
    string word2vec_file;    //the path of the word vector file
    bool has_header;    //the Word2Vec File contain a header or not
    int vector_length;

public:

    Word2VecTool(const string word2vec_file, bool has_header);    //initialize a object with word vector file path 
    void initWordIndexDict();    //initialize word_index_dict with geiven word vector file
    void getWrodVect(string, MatrixXd&);    //get the vector of geiven word
    int getVectorLength();    //the getter function of vector_length
    void saveWordDict(string file_name);    //save the word index dictionary into file
    void loadWordDict(string file_name);    //load the word index dictionary from file
    double calSimilarity(string word_1, string word_2);    //calculate similarity of the two word geiven
    string getMinEditDistenceWord(string word);    //get a word in word vector dictionary which has mimimal edit distence with the geiven word

};

#endif
