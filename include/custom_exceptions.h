#include <exception>
#include <string>
#include <string.h>
#include <cstdio>

using namespace std;

class ill_defined : public exception
{
    char msg_[150];
    virtual const char *what() const throw()
    {
        return msg_;
    }

public:
    ill_defined(const char* m){
        const char* m_base = "The problem is ill defined.";
        strcat(msg_, m_base);
        strcat(msg_, m);
    }
};