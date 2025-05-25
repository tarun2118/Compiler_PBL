#include <iostream>
#include <unordered_map>
#include <vector>
#include <set>
#include <map>
#include <iomanip>
#include <fstream>
using namespace std;

// Function to build and print the precedence table
map<string, map<string, string>> buildPrecedenceTableFromGrammar(const string &filename, const unordered_map<string, vector<vector<string>>> &leadLastMap,
                                                                 const string &startSymbol)
{
    map<string, map<string, string>> table;
    set<string> terminals;

    // Insert $ marker to terminals
    terminals.insert("$");

    // Gather terminals from leadLastMap
    for (const auto &entry : leadLastMap)
    {
        for (const string &t : entry.second[0])
        {
            terminals.insert(t);
        }
        for (const string &t : entry.second[1])
        {
            terminals.insert(t);
        }
    }

    ifstream file(filename);
    string line;

    while (getline(file, line))
    {
        size_t arrowPos = line.find("->");
        if (arrowPos == string::npos)
            continue;

        string lhs = line.substr(0, arrowPos);
        string rhs = line.substr(arrowPos + 2);

        // Analyze RHS symbols for pairs x y
        for (size_t i = 0; i + 1 < rhs.size(); i++)
        {
            string x(1, rhs[i]);
            string y(1, rhs[i + 1]);
            bool xIsNonTerminal = (leadLastMap.find(x) != leadLastMap.end());
            bool yIsNonTerminal = (leadLastMap.find(y) != leadLastMap.end());
            if (i + 2 < rhs.size())
            {
                string z(1, rhs[i + 2]);
                string mid(1, rhs[i + 1]); // the symbol between x and z
                bool zIsNT = (leadLastMap.find(z) != leadLastMap.end());
                bool midIsNT = (leadLastMap.find(mid) != leadLastMap.end());

                if (!xIsNonTerminal && midIsNT && !zIsNT)
                {
                    table[x][z] = ".=";
                }
            }

            // Rule 1: x .= y if both terminals or separated by single non-terminal
            if (!xIsNonTerminal && !yIsNonTerminal)
            {
                table[x][y] = ".=";
            }
            else if (!xIsNonTerminal && yIsNonTerminal)
            {
                // x <. t for every terminal t in LEAD(y)
                for (const string &t : leadLastMap.at(y)[0])
                {
                    // Manually handle + <. *
                    if (x == "+" && t == "*")
                        table[x][t] = "<.";
                    else if (!(table[x][t] == ">." || table[x][t] == ".=")) // don’t override >. or .=
                        table[x][t] = "<.";
                }
            }
            else if (xIsNonTerminal && !yIsNonTerminal)
            {
                // t >. y for every terminal t in LAST(x)
                for (const string &t : leadLastMap.at(x)[1])
                {
                    // Manually handle * >. +
                    if (t == "*" && y == "+")
                        table[t][y] = ">.";
                    else if (!(table[t][y] == "<." || table[t][y] == ".=")) // don’t override <. or .=
                        table[t][y] = ">.";
                }
            }
        }
    }

    // Add rules for $ as end marker
    for (const string &t : leadLastMap.at(startSymbol)[0])
        table["$"][t] = "<.";
    for (const string &t : leadLastMap.at(startSymbol)[1])
        table[t]["$"] = ">.";

    // Print table body
    for (const string &row : terminals)
    {
        cout << setw(6) << row;
        for (const string &col : terminals)
        {
            if (table[row].count(col))
                cout << setw(6) << table[row][col];
            else
                cout << setw(6) << " ";
        }
        cout << "\n";
    }
    return table;
}