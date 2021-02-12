#include<iostream>
#include <iomanip>
#include<vector>
#include "prettyprint.hpp"
#include "vectorNd.hpp"

#include<numeric>

#include <lapacke.h>
#include <cblas.h>

using namespace VecMat;

int main(){

    //without argument zero vector is made    

    class matrix<3> vv11(1.13000000001);
    vv11[0][0]=-1.3;
    vv11[0][1]=-5.0839402;
    vv11[1][1]=5.03048723;
    vv11[2][1]=20.3046;
    vv11[2][2]=-20.3046;

    //You can provde the double array to give arbitral initial values.
    std::cout << "pow(v11, 100)[0][0]" << pow(vv11, 11)  <<std::endl;
    //std::cout << "pow(v11, 100)[0][0]" << pow_v2(vv11, 11)  <<std::endl;



}
