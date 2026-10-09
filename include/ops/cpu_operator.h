#ifndef CPU_OPERATOR_H
#define CPU_OPERATOR_H

#include "ops/operator_base.h"


// Automatic inherant parent's pure virtual function.
class CpuOperator : public Operator{
    public:
        ~CpuOperator() override = default;

    protected:

};


#endif