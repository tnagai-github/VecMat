#include<iostream>
#include <iomanip>
#include<vector>
#include "prettyprint.hpp"
#include "vecNd.hpp"
#include "vec.hpp"

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

    class matNd<4> I5;
    I5=I4;


    // examples of vector calculations
    std::cout <<std::fixed;
    std::cout << "vecNd<VDIM>"              <<std::endl;
    std::cout << "VDIM =3"              <<std::endl;
    std::cout << "I4 == I4:" << (I4==I4)             <<std::endl;
    std::cout << "I4 == I4+I4:" << (I4 == (I4+I4))   <<std::endl;
    std::cout << "vv12:" << vv12             <<std::endl;
    std::cout << "vv11:" << vv11             <<std::endl;
    std::cout << "vv12:" << vv12             <<std::endl;
    std::cout << "vv13:" << vv13             <<std::endl;
    std::cout << "vv14:" << vv14             <<std::endl;
    std::cout << "+vv14:" << +vv14             <<std::endl;
    std::cout << "-vv14:" << -vv14             <<std::endl;
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
    std::cout << "vv23.at(3) :" << vv23.at(3)  <<std::endl;
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
    std::cout << "+mat14:" << +mat14            <<std::endl;
    std::cout << "-mat14:" << -mat14            <<std::endl;
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
    std::cout << "rot_by_z(45 deg) *  [1, 0, 0] : " << rot_by_z(45/360.*2*M_PI)*vecNd<3>{1,0,0} <<std::endl;
    std::cout << "rot_by_z(45 deg) *  [0, 1, 0] : " << rot_by_z(45/360.*2*M_PI)*vecNd<3>{0,1,0} <<std::endl;
    std::cout << "rot_by_x(45 deg) *  [0, 1, 0] : " << rot_by_x(45/360.*2*M_PI)*vecNd<3>{0,1,0} <<std::endl;
    std::cout << "rot_by_x(45 deg) *  [0, 0, 1] : " << rot_by_x(45/360.*2*M_PI)*vecNd<3>{0,0,1} <<std::endl;
    std::cout << "rot_by_y(45 deg) *  [0, 0, 1] : " << rot_by_y(45/360.*2*M_PI)*vecNd<3>{0,0,1} <<std::endl;
    std::cout << "rot_by_y(45 deg) *  [1, 0, 0] : " << rot_by_y(45/360.*2*M_PI)*vecNd<3>{1,0,0} <<std::endl;
    std::cout << "rot_mat(45/360.*2*M_PI): " << rot_mat(45/360.*2*M_PI) <<std::endl;
    std::cout << "rot_mat(45 deg) * [1, 0]: " << rot_mat(45/360.*2*M_PI)*vecNd<2>{1,0}  <<std::endl;
    std::cout << "rot_mat(45 deg) * [0, 1]: " << rot_mat(45/360.*2*M_PI)*vecNd<2>{0,1}  <<std::endl;
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

    std::cout << " *wrapper's results(dgeev)*" << std::endl;
    std::cout << ans2.eigenValRe << std::endl;
    std::cout << ans2.eigenValIm << std::endl;
    std::cout << ans2.eigenVecl << std::endl;
    std::cout << ans2.eigenVecr << std::endl;
    std::cout << ans2 << std::endl;
    std::cout << "***********" << std::endl;


    double init_array3[4][4]= {{2,5,0,1},{5,5,1,1},{0,1,4,0},{1,1,0,1}};
    class matNd<4> mat24(init_array3);
    std::cout << "mat24" << mat24 << std::endl;

    std::cout << "***********" << std::endl;
    std::cout << " *is_symetrici(mat24):  " << is_symetric(mat24) << std::endl;
    std::cout << " *is_positive_difinite_dsy(mat24):  " << is_positive_difinite_dsy(mat24) << std::endl;
    std::cout << " *calc_inverse(mat24):  "  << calc_inverse(mat24) << std::endl;
    std::cout << " *mat24*calc_inverse(mat24):  "  << mat24*calc_inverse(mat24) << std::endl;
    std::cout << "***********" << std::endl;

    ans2 = wrap_dgeev(mat24);
    std::cout << " *wrapper's results(dgeev, mat24)*" << std::endl;
    std::cout << ans2.eigenValRe << std::endl;
    std::cout << ans2.eigenValIm << std::endl;
    std::cout << ans2.eigenVecl << std::endl;
    std::cout << ans2.eigenVecr << std::endl;
    std::cout << ans2 << std::endl;
    std::cout << "***********" << std::endl;


    ans2 = wrap_dsyev(mat24);
    std::cout << " *wrapper's results(dsyev, mat24)*" << std::endl;
    std::cout << ans2.eigenValRe << std::endl;
    std::cout << ans2.eigenValIm << std::endl;
    std::cout << ans2.eigenVecl << std::endl;
    std::cout << ans2.eigenVecr << std::endl;
    std::cout << ans2 << std::endl;
    std::cout << "***********" << std::endl;
    std::cout << "*conform *" << std::endl;
    std::cout << ans2.eigenVecl*mat24*ans2.eigenVecr.T() << std::endl;
    std::cout << ans2.QT()*mat24*ans2.Q() << std::endl;
    std::cout << "***********" << std::endl;
    
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



    Vec vecA(3,1.0/sqrt(3.0));
    Vec vecB(std::vector<double>{1,-1,1});
    vecA[1] = 2;
    Vec vecC(vecA);

    std::cout << "vecA          " << vecA           << std::endl;
    std::cout << "vecA          " << vecA           << std::endl;
    std::cout << "vecA.dot(vecA)" << vecA.dot(vecA) << std::endl;
    std::cout << "vecA.abs()    " << vecA.abs()     << std::endl;
    std::cout << "vecA.sum()    " << vecA.sum()     << std::endl;
    std::cout << "vecA*vecB     " << vecA*vecB      << std::endl;
    std::cout << "(vecB)        " << (vecB)         << std::endl;
    std::cout << "(vecB*=5)     " << (vecB*=5)      << std::endl;
    std::cout << "(vecB/=5)     " << (vecB/=5)      << std::endl;
    std::cout << "(vecB-=vecA)  " << (vecB-=vecA)   << std::endl;
    std::cout << "(+vecB)       " << (+vecB)        << std::endl;
    std::cout << "(-vecB)       " << (-vecB)        << std::endl;

    Mat matA(std::vector<std::vector<double>>{std::vector<double>{1,2,3}, std::vector<double>{4,5,6}, std::vector<double>{7,8,9}, std::vector<double>{10,11,12}});

    Mat matB(std::vector<std::vector<double>>{std::vector<double>{1,1,1}, std::vector<double>{2,2,2}, std::vector<double>{3,3,3}, std::vector<double>{12,11,10}});
    Mat matC(matB);
    Mat matD=(matB);
    //Mat matE=(std::vector<std::vector<double>>{std::vector<double>{1,1,1}, std::vector<double>{2,3,2}, std::vector<double>{3,3,4}});
    Mat matE=(std::vector<std::vector<double>>{std::vector<double>{1,1}, std::vector<double>{1,2}});
    matD[1][2]+=100;
    std::cout << "matA          " << matA           << std::endl;
    std::cout << "matB          " << matB           << std::endl;
    std::cout << "matC          " << matC           << std::endl;
    std::cout << "matC[0][2]    " << matC[0][2]    << std::endl;
    matC[0][2] = 999 ;
    std::cout << "matC[0][2]    " << matC[0][2]    << std::endl;
    std::cout << "matC          " << matC          << std::endl;
    std::cout << "matD          " << matD          << std::endl;
    std::cout << "matD+=matC    " << (matD+=matC)    << std::endl;
    std::cout << "+matD         " << (+matD)    << std::endl;
    std::cout << "-matD         " << (-matD)    << std::endl;
    std::cout << "matD.T()         " << (matD.T())    << std::endl;
    std::cout << "matD*matD.T()    " << (matD*matD.T())    << std::endl;
    std::cout << "matD*matD.T()    " << (matD*matD.T())    << std::endl;
    std::cout << "matD*vecB        " << (matD*vecB)    << std::endl;
    std::cout << "vecB*matD        " << (vecB*matD.T())    << std::endl;
    std::cout << "matD/100.0       " << (matD/100.0)    << std::endl;
    std::cout << "matD*10.0        " << (matD*10.0)    << std::endl;
    std::cout << "10.0*matD*10.0  " << (10.0*matD*10.0)    << std::endl;
    std::cout << "matE             " <<matE           << std::endl;
    std::cout << "matE.trace() " <<matE.trace()   << std::endl;
    std::cout << "matE.triangle() " <<matE.triangle()   << std::endl;
    std::cout << "matE.det()      " <<matE.det()   << std::endl;

    std::cout << "**************************"    << std::endl;
#ifdef vecNd_BLAS
    auto ans3=wrap_dgeev(matE) ;
    std::cout << "Eigen vectors \n " << ans3   << std::endl;
   
    //check
    std::cout << "Left vectors"    << std::endl;
    for(size_t i = 0; i<matE.mm.size(); i++){
      std::cout << " " << (ans3.eigenVeclv[i]*ans3.eigenValRe[i])   << std::endl;
      std::cout << " " << (ans3.eigenVeclv[i]*matE)   << std::endl;
    }
    std::cout << "Right vectors"    << std::endl;
    for(size_t i = 0; i<matE.mm.size(); i++){
      std::cout << " " << (ans3.eigenVecrv[i]*ans3.eigenValRe[i])   << std::endl;
      std::cout << " " << (matE*ans3.eigenVecrv[i])   << std::endl;
    }
#endif
}
