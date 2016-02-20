/*************************************************************************
	> File Name: string_tool.cpp
	> Author: Wang Junqi
	> Mail: 351868656@qq.com 
	> Created Time: 2016年01月11日 星期一 19时33分29秒
 ************************************************************************/

#include"../include/string_tool.h"

//split the string with geiven delim
void StringTool::split(const string &str, string delim, vector<string> &result)
{
    /*
     * argument : str -- the original string
     *            delim -- the separator set used to split the string 
     *            result -- result vector to store the substring splited by delim
     */
    
    int last_index = 0;    //the last pose of the delim
    int current_index = 0;    //the current pose of the delim

    //find all delim in the string
    while ( (current_index = str.find_first_of(delim, last_index)) != string::npos)
    {
        string temp_str = str.substr(last_index, current_index - last_index);
        result.push_back(temp_str);
        last_index = current_index + 1;
    }

    //the string ends with delim or not
    if (last_index != str.size())
    {
        string temp_str = str.substr(last_index);    //substr form last_index to the end
        result.push_back(temp_str);
    }
    else
    {
        result.push_back("");
    }

    return;
}

//calculate the edit distence between two string
int StringTool::calEditDistence(const string &word_1, const string &word_2)
{
    /*
     * argument : word_1, word_2 -- the two string that will be calculate edit distence
     * return : the edit distence between the two string
     */
    
    int word_1_length = word_1.size();
    int word_2_length = word_2.size();
    
    //initialize result vector
    vector<int> result(word_1_length + 1, 0);
    for (int i = 0; i <= word_1_length; ++i)
    {
        result[i] = i;
    }
    
    //calculate edit distence with DP
    for (int word_2_index = 0; word_2_index < word_2_length; ++word_2_index)
    {
        int last_dis = word_2_index;
        result[0] = word_2_index + 1;

        for (int word_1_index = 1; word_1_index <= word_1_length; ++word_1_index)
        {
            if (word_1[word_1_index - 1] == word_2[word_2_index])
            {
                int temp_last_dis = last_dis;
                last_dis = result[word_1_index];
                result[word_1_index] = temp_last_dis;
            }
            else
            {
                //three opperation stand for change a char, add a char and delete a char in word_2
                int change_dis = last_dis + 1;
                int add_dis = result[word_1_index - 1] + 1;
                int del_dis = result[word_1_index] + 1;

                last_dis = result[word_1_index];
                
                //find the minimal value among the three value
                int min_val = (change_dis < add_dis) ? change_dis : add_dis;
                min_val = (min_val < del_dis) ? min_val : del_dis;

                result[word_1_index] = min_val;
            }
        }
    }

    return result[word_1_length];
}

//tokenize the sentence into a vector with the geiven delimitation
vector<string> StringTool::tokenize(const string sentence, string delimitation)
{
    /*
     * argument : sentence -- the sentence to be tokenized
     *            delimitation -- candidate char used to split and tokenize the sentence
     *                            the default of the delimitation is " ,.?!()[]{}<>~#&*`\'\";"
     * return : the vector where tokenized words stored 
     */ 
    vector<string> result;
    
    int last_index = 0;    //the last pose of the delimitation
    int current_index = 0;    //the current pose of the delimitation

    //find all delimitation in the string
    while ( (current_index = sentence.find_first_of(delimitation, last_index)) != string::npos)
    {
        string temp_str = sentence.substr(last_index, current_index - last_index);
        if (temp_str.size() > 0)
        {
            result.push_back(temp_str);
        }
        string current_delim = "";
        current_delim.push_back(sentence[current_index]);
        if (current_delim != " ")
        {
            result.push_back(current_delim);
        }
        last_index = current_index + 1;
    }

    //the string ends with delimitation or not
    if (last_index != sentence.size())
    {
        string temp_str = sentence.substr(last_index);    //substr form last_index to the end
        result.push_back(temp_str);
    }

    return result;
}

/*
int main()
{
    StringTool st;
    vector<string> result = st.tokenize("i love you, my love! and i'm not good love-man. and...you're my \"good-one\".");
    for (int i = 0 ; i < result.size(); ++i)
    {
        cout << result[i] << endl;
    }
    return 0;
}
*/
