/*************************************************************************
	> File Name: word2vec_tool.cpp
	> Author: Wang Junqi
	> Mail: 351868656@qq.com 
	> Created Time: 2016年01月08日 星期五 21时44分24秒
 ************************************************************************/

#include"../include/word2vec_tool.h"
#include<cstdio>
#include<algorithm>


using namespace Eigen;

//Initialize the object with word vector file path
Word2VecTool::Word2VecTool(const string word2vec_file, bool has_header = false)
{
    FILE *w2v_p = fopen(word2vec_file.c_str(), "rb");
    //check the file exist or not
    if (w2v_p == NULL)
    {
        cout << word2vec_file << " file does not exist!" << endl;
        return;
    }
    
    //skip the header of the file
    if (has_header)
    {
        while (fgetc(w2v_p) != '\n')
        {
            ;
        }
    }

    //skik the word at the head of line
    while (fgetc(w2v_p) != ' ')
    {
        ;
    }

    int vector_length = 0;
    double __wights;
    
    //count vector length
    do 
    {
        fscanf(w2v_p, "%lf", &__wights);
        ++vector_length;
    } while (fgetc(w2v_p) != '\n');

    this -> word2vec_file = word2vec_file;
    this -> has_header = has_header;
    this -> vector_length = vector_length;

    fclose(w2v_p);

    return;
}

//Initialize word_index_dict with geiven word2vec file
void Word2VecTool::initWordIndexDict()
{
    FILE *w2v_p = fopen(word2vec_file.c_str(), "rb");    //open the file with the mode 'binary'

    char *temp_word = new char[10000];
    double temp_weight;
    long long now_pos = 0;    //the word offset in the word2vec file

    //cut off the header of the word vector file
    if (has_header)
    {
        while (fgetc(w2v_p) != '\n')
        {
            ;
        }
        now_pos += ftell(w2v_p);
    }

    while (true)
    {
        fscanf(w2v_p, "%s", temp_word);    //read the word
        word_index_dict.insert(map<string, long long>::value_type(temp_word, now_pos));     //add the word into word2vec dict
        char flag;
        //if file_pointer meet the end of a line or EOF , break the recurrent
        while ((flag = fgetc(w2v_p)) != '\n' && flag != EOF)
        {
            fscanf(w2v_p, "%lf", &temp_weight);
        }
        if (flag == EOF)
        {
            break;
        }
        else
        {
            now_pos = ftell(w2v_p);    //record the offset
            //print out the word_index_dict
            if (word_index_dict.size() % 10000 == 0)
            {
                cout << word_index_dict.size() << endl; 
            }
        }
    }

    fclose(w2v_p);

    return;
}

void Word2VecTool::getWrodVect(string word, MatrixXd &word_vector)
{
    /*
     * argument : word -- the word
     *            word_vector -- a col vector of the word
     */
        
    transform(word.begin(), word.end(), word.begin(), towlower);    //transform Uper to Lower

    FILE *w2v_p = fopen(word2vec_file.c_str(), "rb");    //open the file with the mode 'binary'

    //word_vec dict contain the word or not
    if (word_index_dict.find(word) == word_index_dict.end())
    {
        cout << '\'' << word << '\'' << " dose not exist in the word2vector file !" << endl;
        
        //if '-' exists in the word
        if ()
        {
        
        }
        else
        {
            int min_dis = 10000;
            for ()
        }
    }
    else
    {
        long long word_offset = word_index_dict[word];
    
        fseek(w2v_p, word_offset, 0);    //seek the pointer to the word offset relative to the head of file
    
        //skip the word at the head of the line
        while (fgetc(w2v_p) != ' ')
        {
            ;
        }
        
        //read word vector from current file pointer
        for (int row_num = 0; row_num < word_vector.rows(); ++row_num)
        {
            fscanf(w2v_p, "%lf", &word_vector(row_num, 0));
        }
    }

    fclose(w2v_p);

    return;
}

//the getter function of vector_length
int Word2VecTool::getVectorLength()
{
    return vector_length;
}

int main()
{
    Word2VecTool wt("../res/glove.42B.300d.txt");
    wt.initWordIndexDict();
    string word;
    
    while (cin >> word)
    {
        MatrixXd temp_vec = MatrixXd::Zero(wt.getVectorLength(), 1);
        wt.getWrodVect(word, temp_vec);
        cout << temp_vec << endl;
    }
    
    /*
    MatrixXd temp_vec = MatrixXd::Zero(wt.getVectorLength(), 1);
    wt.getWrodVect("...", temp_vec);
    cout << wt.getVectorLength() << endl;
    return 0;
    */
}
