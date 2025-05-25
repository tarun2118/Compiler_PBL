// leadlast.cpp
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <set>
#include <cctype>
using namespace std;

bool isNonTerminal(char ch)
{
    return isupper(ch);
}

unordered_map<string, vector<string>> grammar;

// Parse grammar from file
void parseGrammar(const string &filename)
{
    ifstream infile(filename);
    string line;
    while (getline(infile, line))
    {
        if (line.empty()) continue;

        size_t pos = line.find("->");
        if (pos == string::npos) continue;

        string lhs = line.substr(0, pos);
        string rhs = line.substr(pos + 2);

        grammar[lhs].push_back(rhs);
    }
}

// Calculate LEAD sets
unordered_map<string, set<string>> calculateLeadSets()
{
    unordered_map<string, set<string>> lead;

    for (auto &g : grammar)
        lead[g.first] = set<string>();

    bool changed = true;
    while (changed)
    {
        changed = false;
        for (auto &g : grammar)
        {
            const string &A = g.first;
            for (const string &prod : g.second)
            {
                if (prod.empty()) continue;

                if (!isNonTerminal(prod[0]))
                {
                    if (lead[A].insert(string(1, prod[0])).second)
                        changed = true;
                }
                else
                {
                    char B = prod[0];
                    for (const string &s : lead[string(1, B)])
                    {
                        if (lead[A].insert(s).second)
                            changed = true;
                    }

                    //If prod[1] is terminal and prod[0] is non-terminal, add prod[1] to lead[A]
                    if (prod.size() >= 2 && !isNonTerminal(prod[1]))
                    {
                        if (lead[A].insert(string(1, prod[1])).second)
                            changed = true;
                    }
                }
            }
        }
    }
    return lead;
}

// Calculate LAST sets
unordered_map<string, set<string>> calculateLastSets()
{
    unordered_map<string, set<string>> last;

    for (auto &g : grammar)
        last[g.first] = set<string>();

    bool changed = true;
    while (changed)
    {
        changed = false;
        for (auto &g : grammar)
        {
            const string &A = g.first;
            for (const string &prod : g.second)
            {
                if (prod.empty()) continue;

                char lastChar = prod[prod.size() - 1];
                if (!isNonTerminal(lastChar))
                {
                    if (last[A].insert(string(1, lastChar)).second)
                        changed = true;
                }
                else
                {
                    char B = lastChar;
                    for (const string &s : last[string(1, B)])
                    {
                        if (last[A].insert(s).second)
                            changed = true;
                    }

                    //If prod[prod.size()-2] is terminal and last is non-terminal
                    if (prod.size() >= 2 && !isNonTerminal(prod[prod.size() - 2]))
                    {
                        if (last[A].insert(string(1, prod[prod.size() - 2])).second)
                            changed = true;
                    }
                }
            }
        }
    }
    return last;
}

// Calculate LEAD and LAST sets and return in unordered_map
unordered_map<string, vector<vector<string>>> calculateLeadLast(const string &filename)
{
    grammar.clear(); // clear previous data if any
    parseGrammar(filename);

    auto leadSets = calculateLeadSets();
    auto lastSets = calculateLastSets();

    unordered_map<string, vector<vector<string>>> mp;

    for (auto &entry : grammar)
    {
        string nonTerminal = entry.first;
        //Converts the set<string>into a vector<string> so you can store them as indexed sequences.
        vector<string> leadVec(leadSets[nonTerminal].begin(), leadSets[nonTerminal].end());
        vector<string> lastVec(lastSets[nonTerminal].begin(), lastSets[nonTerminal].end());

        mp[nonTerminal] = vector<vector<string>>(2);
        mp[nonTerminal][0] = leadVec;
        mp[nonTerminal][1] = lastVec;
    }
    return mp;
}
