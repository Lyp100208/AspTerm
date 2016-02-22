/*************************************************************************
	> File Name: string_tool.h
	> Author: Wang Junqi
	> Mail: 351868656@qq.com 
	> Created Time: 2016年01月11日 星期一 15时18分26秒
 ************************************************************************/

#ifndef STRING_TOOL_H
#define STRING_TOOL_H

#include<iostream>
#include<vector>

using namespace std;

class StringTool
{

public:

    void split(const string&, string, vector<string>&);    //split a string with geiven delim
    int calEditDistence(const string&, const string&);    //calculate the edit distence between two wrod
    vector<string> tokenize(const string, string delimitation = " ,.?!()[]{}<>~#&*`\";");    //tokenize the sentence into a vector with the geiven delimitation

};
  
#endif
