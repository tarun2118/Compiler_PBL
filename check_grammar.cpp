#include <iostream>
#include <fstream>
#include <cctype>
using namespace std;

// Check if a character is a non-terminal (A-Z)
bool isNon_Terminal(char ch)
{
    return isupper(ch);
}

// Function to check grammar validity for operator precedence
bool isOperatorPrecedenceGrammar(const string &filename)
{
    ifstream infile(filename);
    string line;

    while (getline(infile, line))
    {
        if (line.empty())
            continue;

        size_t pos = line.find("->");
        if (pos == string::npos)
        {
            return false; // invalid production format
        }

        string lhs = line.substr(0, pos);
        string rhs = line.substr(pos + 2);

        if (rhs == "@")
        {
            return false;
        }

        for (size_t i = 0; i + 1 < rhs.size(); ++i)
        {
            if (isNon_Terminal(rhs[i]) && isNon_Terminal(rhs[i + 1]))
            {
                return false;
            }
        }
    }

    return true;
}
