#include<iostream>
#include <iomanip>
#include<vector>
#include "prettyprint.hpp"
#include "vecNd.hpp"

#include<numeric>

#ifdef vecNd_BLAS
#include <lapacke.h>
#include <cblas.h>
#endif

using namespace VecMat;

int main(){

    //without argument zero vector is made    
    class vecNd<3> vv10;
    std::cout << "vv10:" << vv10             <<std::endl;

    //with one double argument, all components will be initialized by the value specified.
    class vecNd<3> vv11(1.0);
    class vecNd<3> vv12(2.0);

    //You can provde the double array to give arbitral initial values.
    double darray1[3] = {0.5, 1, 2.0};
    class vecNd<3> vv13(darray1);
    class vecNd<3> vv14 {1,2,100};

    // the same applies for different dimensions. 
    class vecNd<4> vv21(1.0);
    class vecNd<4> vv22(2.0);
    double darray2[4] = {0.5, 1, 2.0, -1.0};
    class vecNd<4> vv23(darray2);

    class matNd<3> I3;
    class matNd<4> I4;

    // examples of vector calculations
    std::cout <<std::fixed;
    std::cout << "vecNd<VDIM>"              <<std::endl;
    std::cout << "VDIM =3"              <<std::endl;
    std::cout << "vv11:" << vv11             <<std::endl;
    std::cout << "vv12:" << vv12             <<std::endl;
    std::cout << "vv13:" << vv13             <<std::endl;
    std::cout << "vv14:" << vv14             <<std::endl;
    std::cout << "vv13[0] :" << vv13[0]  <<std::endl;
    std::cout << "vv13[1] :" << vv13[1]  <<std::endl;
    std::cout << "vv13[2] :" << vv13[2]  <<std::endl;
    std::cout << "Inner-product"              <<std::endl;
    std::cout << " vv13*vv12     :" << vv13*vv12        <<std::endl;
    std::cout << " dot(vv13,vv12):" << dot(vv13,vv12)   <<std::endl;
    std::cout << " vv13.dot(vv12):" << vv13.dot(vv12)   <<std::endl;
    std::cout << "abs"              <<std::endl;
    std::cout << " vv11.abs() :" << vv11.abs()      <<std::endl;
    std::cout << " abs(vv11)  :" << abs(vv11)        <<std::endl;
    std::cout << "(vv11+vv13):" << (vv11+vv13) <<std::endl;
    std::cout << "(vv11-vv13):" << (vv11-vv13) <<std::endl;
    std::cout << "(20.0*vv11*10.0):" << (20.0*vv11*10.0) <<std::endl;
    std::cout << "(vv11/10.0):" << (vv11/10.0) <<std::endl;
    std::cout << "(vv11+=vv13) :" << (vv11+=vv13)  <<std::endl;
    std::cout << "(vv11-=vv13) :" << (vv11-=vv13)  <<std::endl;
    std::cout << "(vv11*=10.0) :" << (vv11*=10.0)  <<std::endl;
    std::cout << "cross product (three dimension only)"          << std::endl;
    std::cout << "cross(vv13,vv12): " << cross(vv13,vv12) <<std::endl;
    std::cout << "***********************" <<std::endl;


    std::cout <<               std::endl;
    std::cout << "VDIM =4"              <<std::endl;
    std::cout << "vv21:" << vv21             <<std::endl;
    std::cout << "vv22:" << vv22             <<std::endl;
    std::cout << "vv23:" << vv23             <<std::endl;
    std::cout << "vv23[0] :" << vv23[0]  <<std::endl;
    std::cout << "vv23[1] :" << vv23[1]  <<std::endl;
    std::cout << "vv23[2] :" << vv23[2]  <<std::endl;
    std::cout << "vv23[3] :" << vv23[3]  <<std::endl;
    std::cout << "Inner-product"              <<std::endl;
    std::cout << " vv23*vv22     :" << vv23*vv22        <<std::endl;
    std::cout << " dot(vv23,vv22):" << dot(vv23,vv22)   <<std::endl;
    std::cout << " vv13.dot(vv22):" << vv23.dot(vv22)   <<std::endl;
    std::cout << "abs"              <<std::endl;
    std::cout << " vv21.abs() :" << vv21.abs()      <<std::endl;
    std::cout << " abs(vv21)  :" << abs(vv21)        <<std::endl;
    std::cout << "(vv21+vv23):" << (vv21+vv23) <<std::endl;
    std::cout << "(vv21-vv23):" << (vv21-vv23) <<std::endl;
    std::cout << "(20.0*vv21*10.0):" << (20.0*vv21*10.0) <<std::endl;
    std::cout << "(vv21/10.0):" << (vv21/10.0) <<std::endl;
    std::cout << "(vv21+=vv23) :" << (vv21+=vv23)  <<std::endl;
    std::cout << "(vv21-=vv23) :" << (vv21-=vv23)  <<std::endl;
    std::cout << "(vv21*=10.0) :" << (vv21*=10.0)  <<std::endl;
    std::cout << "***********************" <<std::endl;


    // without initializer, the unit matrix will be made
    class matNd<3> mat11;
    // if one double value is supplied, all compoments will be initialized by the value.
    class matNd<3> mat12(1);
    // you can give double array[VDIM][VDIM] to make arbitraly initialization
    double init_array1[3][3]= {{0,1,1.2},{5.4,0,0.9},{2.4,1.3,0}};
    class matNd<3> mat13(init_array1);
    class matNd<3> mat14 {9,8,7,6,5,4,3,2,1};

    class matNd<4> mat21;
    class matNd<4> mat22(1);
    //double init_array2[4][4]= {{-1,2,3,4},{5,-1,7,8},{9,10,0,12},{13,14,15,1}};
    double init_array2[4][4]= {{2,0,0,1},{0,5,1,0},{-2,0,4,0},{0,1,1,1}};
    class matNd<4> mat23(init_array2);



    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "***************************"              <<std::endl;
    std::cout << "matNd<VDIM>"              <<std::endl;
    std::cout << "VDIM =3"                    <<std::endl;
    std::cout << "mat11:" << mat11            <<std::endl;
    std::cout << "mat12:" << mat12            <<std::endl;
    std::cout << "mat13:" << mat13            <<std::endl;
    std::cout << "mat14:" << mat14            <<std::endl;
    std::cout << "mat11[1][2]:" <<mat11[1][2]      <<std::endl;
    std::cout << "mat11[2][2]:" <<mat11[2][2]      <<std::endl;
    std::cout << "(mat11+=mat12):" << (mat11+=mat12)   <<std::endl;
    std::cout << "(mat11-=mat12):" << (mat11-=mat12)   <<std::endl;
    std::cout << "(mat11+mat12):" << (mat11+mat12)   <<std::endl;
    std::cout << "(mat11-mat12):" << (mat11-mat12)   <<std::endl;
    std::cout << "mat11*=2.0: "   << (mat11*=2.0)     <<std::endl;
    std::cout << "mat11/=2.0: "   << (mat11/=2.0)     <<std::endl;
    std::cout << "mat11*2.0: "    << (mat11*2.0)     <<std::endl;
    std::cout << "mat11/2.0: "    << (mat11/2.0)     <<std::endl;
    std::cout << "mat13: "        <<  (mat13)         <<std::endl;
    std::cout << "mat13.T(): "    << (mat13.T())      <<std::endl;
    std::cout << "mat13*mat13: "    << (mat13*mat13)      <<std::endl;
    std::cout << "***************************"              <<std::endl;
    std::cout << "product of matrix and vector is also possible"  <<std::endl;
    std::cout << "mat13*vv12:"    << (mat13*vv12)     <<std::endl;
    std::cout << "v12*mat13.T()"  << (vv12*mat13.T()) <<std::endl;

    std::cout << "***************************"              <<std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "VDIM =4"              <<std::endl;
    std::cout << "mat21:" << mat21            <<std::endl;
    std::cout << "mat22:" << mat22            <<std::endl;
    std::cout << "mat23:" << mat23            <<std::endl;
    std::cout << "mat21[1][2]:"  << mat21[1][2]      <<std::endl;
    std::cout << "mat21[2][2]:"  << mat21[2][2]      <<std::endl;
    std::cout << "(mat21+=mat22):" << (mat21+=mat22)   <<std::endl;
    std::cout << "(mat21-=mat22):" << (mat21-=mat22)   <<std::endl;
    std::cout << "(mat21+mat22):" << (mat21+mat22)   <<std::endl;
    std::cout << "(mat21-mat22):" << (mat21-mat22)   <<std::endl;
    std::cout << "mat21*=2.0: "   << (mat21*=2.0)     <<std::endl;
    std::cout << "mat21/=2.0: "   << (mat21/=2.0)     <<std::endl;
    std::cout << "mat21*2.0: "    << (mat21*2.0)     <<std::endl;
    std::cout << "mat21/2.0: "    << (mat21/2.0)     <<std::endl;
    std::cout << "mat23: "        <<  (mat23)         <<std::endl;
    std::cout << "mat23.T(): "    << (mat23.T())      <<std::endl;
    std::cout << "mat23*mat23: "    << (mat23*mat23)      <<std::endl;
    std::cout << "***************************"              <<std::endl;
    std::cout << "product of matrix and vector is also possible"  <<std::endl;
    std::cout << "mat23*vv22:"    << (mat23*vv22)     <<std::endl;
    std::cout << "v22*mat23.T():"  << (vv22*mat23.T()) <<std::endl;
    std::cout << "***************************"              <<std::endl;




    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "***************************"              <<std::endl;
    std::cout << "Some rotation matrices are predefined"   <<std::endl;
    std::cout << "rot_by_x(45/360.*2*M_PI): " << rot_by_x(45/360.*2*M_PI) <<std::endl;
    std::cout << "rot_by_y(45/360.*2*M_PI): " << rot_by_y(45/360.*2*M_PI) <<std::endl;
    std::cout << "rot_by_z(45/360.*2*M_PI): " << rot_by_z(45/360.*2*M_PI) <<std::endl;
    std::cout << "***************************"              <<std::endl;




    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "***************************"              <<std::endl;
    std::cout << "Very basic matrix operation"   <<std::endl;
    std::cout << "triangle(mat13): " << triangle(mat13) <<std::endl;
    std::cout << "(mat13.triangle()): " << (mat13.triangle()) <<std::endl;
    std::cout << "det(mat13): " << det(mat13) <<std::endl;
    std::cout << "(mat13.det()): " << (mat13.det()) <<std::endl;
    std::cout << "triangle(mat23): " << triangle(mat23) <<std::endl;
    std::cout << "(mat23.triangle()): " << (mat23.triangle()) <<std::endl;
    std::cout << "det(mat23): " << det(mat23) <<std::endl;
    std::cout << "(mat23.det()): " << (mat23.det()) <<std::endl;
    std::cout << "(pow(mat23,0): " << (pow(mat23,0)) <<std::endl;
    std::cout << "(pow(mat23,1): " << (pow(mat23,1)) <<std::endl;
    std::cout << "(pow(mat23,2): " << (pow(mat23,2)) <<std::endl;
    std::cout << "***************************"       <<std::endl;



// as operators are overloaded, STL related function is usable
    std::vector< class matNd<3>> mat_array1(3);

    std::cout << mat_array1 << std::endl;
    
    class matNd<3> init(0.0);
    auto sum1 = std::accumulate(mat_array1.begin()+1, mat_array1.end(), mat_array1[0]);

    std::cout << "sum1: " << sum1 << std::endl;


    class matNd<4> mat23v2(mat23);
    double wi[4];
    double wr[4];

    //double vl[4][4];
    //double vr[4][4];
    class matNd<4> vl2 ;
    class matNd<4> vr2 ;

#ifdef vecNd_BLAS
	//direct call of blas
    LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', 
                         4, mat23v2.p, 4, wr, 
                          wi, vl2.p, 4,  vr2.p, 
                          4 );
	//called with wrapper 
    auto ans2 = wrap_dgeev(mat23);

    std::cout << " *wrapper's results*" << std::endl;
    std::cout << ans2.eigenValRe << std::endl;
    std::cout << ans2.eigenValIm << std::endl;
    std::cout << ans2.eigenVecl << std::endl;
    std::cout << ans2.eigenVecr << std::endl;
    std::cout << ans2 << std::endl;
    std::cout << "***********" << std::endl;

    //std::cout << wr << std::endl;
    
	// to collect the results obtained by a direct call
    class matNd<4> tmp(vr2.T()); 
    class vecNd<4>  eigenv0(tmp[0]);
    class vecNd<4>  eigenv1(tmp[1]);
    class vecNd<4>  eigenv2(tmp[2]);
    class vecNd<4>  eigenv3(tmp[3]);


    std::cout << " *direct call of lapack *" << std::endl;
    std::cout << eigenv0 << std::endl;
    std::cout << eigenv1 << std::endl;
    std::cout << eigenv2 << std::endl;
    std::cout << eigenv3 << std::endl;

    std::cout << "******" << std::endl;

    std::cout << " *check Mv=av **" << std::endl;
    std::cout << mat23*eigenv0 << std::endl;
    std::cout << wr[0]*eigenv0 << std::endl;

    std::cout << mat23*eigenv1 << std::endl;
    std::cout << wr[1]*eigenv1 << std::endl;

    std::cout << mat23*eigenv2 << std::endl;
    std::cout << wr[2]*eigenv2 << std::endl;

    std::cout << mat23*eigenv3 << std::endl;
    std::cout << wr[3]*eigenv3 << std::endl; //return of vr is column major !!


    

    double ans;
    ans = cblas_ddot(vv11.size, vv11.p, 1, vv12.p,1);
//    double cblas_ddot(const int N, const double *X, const int incX,
//                  const douggble *Y, const int incY);
    std::cout << ans << std::endl;
    std::cout << vv11*vv12 << std::endl;
#endif


}
